// Under the board: the piece tray (set-up mode), or the move list, blunder
// warning and chances chart (play mode).
import type { PieceCode } from "../lib";
import { pieceUrl } from "./Board";

const TRAY: PieceCode[][] = [
  ["wK", "wQ", "wR", "wB", "wN", "wP"],
  ["bK", "bQ", "bR", "bB", "bN", "bP"],
];

export function PieceTray({ onGrab }: { onGrab: (piece: PieceCode, x: number, y: number) => void }) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-3">
      {TRAY.map((row) => (
        <div key={row[0]} className="flex gap-1 rounded-xl bg-ink-900/80 p-1.5 ring-1 ring-white/5">
          {row.map((p) => (
            <button
              key={p}
              title="Drag onto the board"
              className="h-11 w-11 cursor-grab rounded-lg transition hover:bg-white/10 active:cursor-grabbing"
              onPointerDown={(e) => {
                e.preventDefault();
                onGrab(p, e.clientX, e.clientY);
              }}
            >
              <img src={pieceUrl(p)} alt={p} draggable={false} className="h-full w-full p-0.5" />
            </button>
          ))}
        </div>
      ))}
    </div>
  );
}

export interface PlayedMove {
  san: string;
  blunder: string | null; // explanation, if this move threw away a result
}

export function MoveList({ moves, firstTurn }: { moves: PlayedMove[]; firstTurn: "w" | "b" }) {
  // Pair moves into numbered rows. If Black moved first, row 1 starts with "...".
  const cells: (PlayedMove | null)[] = firstTurn === "b" ? [null, ...moves] : moves;
  const rows = [];
  for (let i = 0; i < cells.length; i += 2) rows.push([cells[i], cells[i + 1]]);
  const cell = (m: PlayedMove | null | undefined, isLast: boolean) =>
    m === null ? <span className="text-ink-500">…</span>
    : m === undefined ? null
    : (
      <span className={`rounded px-1.5 py-0.5 font-mono ${isLast ? "bg-white/10" : ""} ${m.blunder ? "text-rose-300" : ""}`}>
        {m.san}{m.blunder ? "??" : ""}
      </span>
    );
  if (!moves.length) return <p className="text-sm text-ink-500">No moves yet. Drag or click a piece to move.</p>;
  return (
    <div className="flex max-h-24 flex-wrap gap-x-4 gap-y-1 overflow-y-auto text-sm">
      {rows.map(([a, b], i) => (
        <span key={i} className="flex items-center gap-1">
          <span className="w-6 text-right text-xs text-ink-500">{i + 1}.</span>
          {cell(a, a === moves[moves.length - 1])}
          {cell(b, b === moves[moves.length - 1])}
        </span>
      ))}
    </div>
  );
}

/** White's winning chance (network) after each move, as a small line chart. */
export function ChanceChart({ values }: { values: number[] }) {
  if (values.length < 2) return null;
  const w = 240;
  const h = 56;
  const pts = values.map((v, i) => [(i / (values.length - 1)) * w, h - v * h]);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-300">White's chance (network)</span>
      <svg viewBox={`0 0 ${w} ${h}`} className="h-14 w-60 overflow-visible">
        <line x1="0" x2={w} y1={h / 2} y2={h / 2} className="stroke-ink-700" strokeDasharray="3 3" />
        <path d={`${line} L${w},${h} L0,${h} Z`} className="fill-emerald-300/10" />
        <path d={line} className="fill-none stroke-emerald-300" strokeWidth="2" strokeLinejoin="round" />
        {pts.length > 0 && <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="3" className="fill-emerald-300" />}
      </svg>
    </div>
  );
}
