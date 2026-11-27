// The chessboard: drag or click to move, legal-move dots, last-move and check
// highlights, sliding animations, and a promotion picker. In set-up mode,
// pieces can be dragged anywhere, dragged off the board to remove them, or
// right-clicked to remove; new pieces arrive by dragging from the tray.
import { useEffect, useRef, useState } from "react";
import type { Color, PieceCode, Placement } from "../lib";
import { squareCoords, squareName } from "../lib";

export interface TrayDrag {
  piece: PieceCode;
  x: number;
  y: number;
}

interface Props {
  placement: Placement;
  orientation: Color;
  mode: "setup" | "play";
  turn: Color;
  legalTargets: (from: string) => string[];
  isPromotion: (from: string, to: string) => boolean;
  onMove: (from: string, to: string, promotion?: string) => void;
  onSetupChange: (next: Placement) => void;
  trayDrag: TrayDrag | null;
  onTrayDragEnd: () => void;
  lastMove?: { from: string; to: string } | null;
  checkSquare?: string | null;
  disabled?: boolean;
}

interface Drag {
  piece: PieceCode;
  from: string | "tray";
  x: number;
  y: number;
  startX: number;
  startY: number;
  moved: boolean;
  wasSelected: boolean;
}

interface Placed {
  id: number;
  code: PieceCode;
  sq: string;
}

const PROMOTIONS = ["Q", "R", "B", "N"] as const;

export function pieceUrl(code: PieceCode): string {
  return `/pieces/${code}.svg`;
}

/**
 * Gives every piece a stable id across positions, so React keeps the same
 * element and CSS can slide it to its new square. A piece that stays put keeps
 * its id; a piece that disappears from one square and a same-kind piece that
 * appears elsewhere are matched (nearest first), which covers moves, captures,
 * castling and undo.
 */
function assignIds(prev: Placed[], next: Placement, nextId: { current: number }): Placed[] {
  const unused = [...prev];
  const out: Placed[] = [];
  const pending: [string, PieceCode][] = [];
  for (const [sq, code] of Object.entries(next)) {
    const k = unused.findIndex((u) => u.sq === sq && u.code === code);
    if (k >= 0) out.push(unused.splice(k, 1)[0]);
    else pending.push([sq, code]);
  }
  for (const [sq, code] of pending) {
    const [f, r] = squareCoords(sq);
    let best = -1;
    let bestDist = Infinity;
    unused.forEach((u, k) => {
      if (u.code !== code) return;
      const [uf, ur] = squareCoords(u.sq);
      const d = Math.abs(uf - f) + Math.abs(ur - r);
      if (d < bestDist) {
        bestDist = d;
        best = k;
      }
    });
    if (best >= 0) out.push({ ...unused.splice(best, 1)[0], sq });
    else out.push({ id: nextId.current++, code, sq });
  }
  return out;
}

export function Board(props: Props) {
  const {
    placement, orientation, mode, turn, legalTargets, isPromotion, onMove, onSetupChange,
    trayDrag, onTrayDragEnd, lastMove, checkSquare, disabled,
  } = props;
  const boardRef = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [promo, setPromo] = useState<{ from: string; to: string } | null>(null);
  const [squarePx, setSquarePx] = useState(64);

  // Stable piece ids for animation.
  const placedRef = useRef<Placed[]>([]);
  const prevPlacement = useRef<Placement | null>(null);
  const nextId = useRef(1);
  const noAnim = useRef<string | null>(null); // square whose piece shouldn't slide (just dropped)
  if (prevPlacement.current !== placement) {
    placedRef.current = assignIds(placedRef.current, placement, nextId);
    prevPlacement.current = placement;
  }
  const placed = placedRef.current;
  useEffect(() => {
    noAnim.current = null;
  });

  // Clear selection when the position or mode changes.
  useEffect(() => {
    setSelected(null);
    setPromo(null);
  }, [placement, mode]);

  // Track the square size, for the dragged piece.
  useEffect(() => {
    const el = boardRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSquarePx(el.clientWidth / 8));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const toView = (sq: string): [number, number] => {
    const [f, r] = squareCoords(sq);
    return orientation === "w" ? [f, 7 - r] : [7 - f, r];
  };
  const squareAt = (x: number, y: number): string | null => {
    const el = boardRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    if (x < rect.left || y < rect.top || x >= rect.right || y >= rect.bottom) return null;
    const col = Math.floor(((x - rect.left) / rect.width) * 8);
    const row = Math.floor(((y - rect.top) / rect.height) * 8);
    return orientation === "w" ? squareName(col, 7 - row) : squareName(7 - col, row);
  };

  const targets = mode === "play" && selected ? legalTargets(selected) : [];

  const tryMove = (from: string, to: string): boolean => {
    if (!legalTargets(from).includes(to)) return false;
    if (isPromotion(from, to)) {
      setPromo({ from, to });
      return true;
    }
    onMove(from, to);
    setSelected(null);
    return true;
  };

  // Drop handling, shared by board drags and tray drags.
  const finishDrag = (d: Drag, x: number, y: number) => {
    const sq = squareAt(x, y);
    if (mode === "setup") {
      const next = { ...placement };
      if (d.from !== "tray") delete next[d.from];
      if (sq) next[sq] = d.piece;
      if (sq || d.from !== "tray") {
        noAnim.current = sq;
        onSetupChange(next);
      }
      return;
    }
    if (d.from === "tray") return;
    if (!d.moved) {
      if (d.wasSelected) setSelected(null); // second click on the same piece
      return;
    }
    if (sq && sq !== d.from && legalTargets(d.from).includes(sq)) {
      noAnim.current = sq;
      tryMove(d.from, sq);
    }
  };

  // While dragging, follow the pointer anywhere on the page.
  useEffect(() => {
    if (!drag) return;
    const move = (e: PointerEvent) =>
      setDrag((d) =>
        d && {
          ...d,
          x: e.clientX,
          y: e.clientY,
          moved: d.moved || Math.hypot(e.clientX - d.startX, e.clientY - d.startY) > 4,
        },
      );
    const up = (e: PointerEvent) => {
      setDrag(null);
      if (drag.from === "tray") onTrayDragEnd();
      finishDrag({ ...drag, moved: drag.moved || Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 4 }, e.clientX, e.clientY);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drag?.from, drag?.piece, drag?.moved, placement, mode]);

  // A drag that started in the piece tray.
  useEffect(() => {
    if (trayDrag && !drag) {
      setDrag({
        piece: trayDrag.piece, from: "tray", x: trayDrag.x, y: trayDrag.y,
        startX: trayDrag.x, startY: trayDrag.y, moved: true, wasSelected: false,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trayDrag]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (disabled || promo) return;
    const sq = squareAt(e.clientX, e.clientY);
    if (!sq) return;
    const piece = placement[sq];

    if (mode === "setup") {
      if (!piece) return;
      if (e.button === 2) {
        const next = { ...placement };
        delete next[sq];
        onSetupChange(next);
        return;
      }
      setDrag({ piece, from: sq, x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, moved: false, wasSelected: false });
      return;
    }

    if (e.button !== 0) return;
    // Clicking a highlighted target with a piece selected: move there.
    if (selected && selected !== sq && targets.includes(sq)) {
      tryMove(selected, sq);
      return;
    }
    if (piece && piece[0] === turn) {
      const wasSelected = selected === sq;
      setSelected(sq);
      setDrag({ piece, from: sq, x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, moved: false, wasSelected });
    } else {
      setSelected(null);
    }
  };

  const light = (sq: string) => {
    const [f, r] = squareCoords(sq);
    return (f + r) % 2 === 1;
  };

  const squares = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const sq = orientation === "w" ? squareName(col, 7 - row) : squareName(7 - col, row);
      const isLight = light(sq);
      const isLast = lastMove && (lastMove.from === sq || lastMove.to === sq);
      const isTarget = targets.includes(sq);
      squares.push(
        <div key={sq} className={`relative flex items-center justify-center ${isLight ? "bg-sq-light" : "bg-sq-dark"}`}>
          {isLast && <div className="absolute inset-0 bg-amber-300/40" />}
          {selected === sq && <div className="absolute inset-0 bg-emerald-300/45" />}
          {checkSquare === sq && (
            <div className="absolute inset-0" style={{ background: "radial-gradient(circle, rgba(239,68,68,0.9) 0%, rgba(239,68,68,0.45) 45%, transparent 75%)" }} />
          )}
          {isTarget && (placement[sq] ? <div className="ring" /> : <div className="dot" />)}
          {col === 0 && (
            <span className={`absolute left-1 top-0.5 text-[10px] font-semibold ${isLight ? "text-sq-dark" : "text-sq-light"}`}>{sq[1]}</span>
          )}
          {row === 7 && (
            <span className={`absolute right-1 bottom-0.5 text-[10px] font-semibold ${isLight ? "text-sq-dark" : "text-sq-light"}`}>{sq[0]}</span>
          )}
        </div>,
      );
    }
  }

  return (
    <div className="relative w-full select-none">
      <div
        ref={boardRef}
        className={`relative grid aspect-square w-full grid-cols-8 grid-rows-8 overflow-hidden rounded-xl shadow-2xl shadow-black/50 ring-1 ring-white/10 ${disabled ? "opacity-90" : ""}`}
        onPointerDown={onPointerDown}
        onContextMenu={(e) => e.preventDefault()}
        style={{ touchAction: "none" }}
      >
        {squares}
        {/* Pieces, positioned by transform so they can slide. */}
        <div className="pointer-events-none absolute inset-0">
          {placed.map((p) => {
            const [x, y] = toView(p.sq);
            const hidden = drag && drag.from === p.sq && (drag.moved || mode === "setup");
            return (
              <div
                key={p.id}
                className={`piece absolute h-[12.5%] w-[12.5%] ${noAnim.current === p.sq ? "no-anim" : ""}`}
                style={{ transform: `translate(${x * 100}%, ${y * 100}%)`, opacity: hidden ? 0 : 1 }}
              >
                <img src={pieceUrl(p.code)} alt={p.code} draggable={false} className="h-full w-full p-[4%]" />
              </div>
            );
          })}
        </div>
        {/* Promotion picker. */}
        {promo && (
          <div className="absolute inset-0 z-20 bg-black/40" onPointerDown={(e) => { e.stopPropagation(); setPromo(null); }}>
            {PROMOTIONS.map((t, i) => {
              const [x, y0] = toView(promo.to);
              const y = y0 === 0 ? i : 7 - i;
              const code = `${turn}${t}` as PieceCode;
              return (
                <button
                  key={t}
                  className="pop absolute h-[12.5%] w-[12.5%] rounded-lg bg-ink-800/95 ring-1 ring-white/20 transition hover:bg-emerald-400/30"
                  style={{ left: `${x * 12.5}%`, top: `${y * 12.5}%` }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    onMove(promo.from, promo.to, t.toLowerCase());
                    setPromo(null);
                    setSelected(null);
                  }}
                >
                  <img src={pieceUrl(code)} alt={t} className="h-full w-full p-[8%]" draggable={false} />
                </button>
              );
            })}
          </div>
        )}
      </div>
      {/* The piece being dragged follows the pointer. */}
      {drag && (drag.moved || drag.from === "tray") && (
        <img
          src={pieceUrl(drag.piece)}
          alt=""
          className="pointer-events-none fixed z-50 drop-shadow-2xl"
          style={{ width: squarePx * 1.1, height: squarePx * 1.1, left: drag.x - squarePx * 0.55, top: drag.y - squarePx * 0.55 }}
        />
      )}
    </div>
  );
}
