// Shared types, the analysis API, and small chess helpers.

export type Color = "w" | "b";
export type PieceCode = `${Color}${"K" | "Q" | "R" | "B" | "N" | "P"}`;
export type Placement = Record<string, PieceCode>; // square ("e4") -> piece

export const FILES = "abcdefgh";

export interface PerfectMove {
  result: number; // +1 win, 0 draw, -1 loss for the player making the move
  mate_in: number;
  text: string;
}

export interface MoveRow {
  uci: string;
  san: string;
  perfect: PerfectMove | null;
  network: { win: number; draw: number; loss: number }; // for the mover
  best: boolean;
}

export interface Analysis {
  valid: boolean;
  reason?: string;
  fen?: string;
  turn?: Color;
  pieces?: number;
  network?: { white: number; draw: number; black: number; placeholder: boolean; trained_range: boolean };
  perfect?:
    | { available: true; result: number; mate_in: number; text: string }
    | { available: false; why: string };
  ranked_by?: "perfect" | "network";
  moves?: MoveRow[];
  game_over?: string | null;
}

export interface Status {
  tables: string[];
  network: "trained" | "placeholder";
}

export async function analyze(fen: string, signal?: AbortSignal): Promise<Analysis> {
  const res = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fen }),
    signal,
  });
  if (!res.ok) throw new Error(`analysis failed (${res.status})`);
  return res.json();
}

export async function fetchStatus(): Promise<Status> {
  const res = await fetch("/api/status");
  if (!res.ok) throw new Error("server not reachable");
  return res.json();
}

export async function randomFen(material?: string): Promise<string> {
  const q = material ? `?material=${encodeURIComponent(material)}` : "";
  const res = await fetch(`/api/random${q}`);
  if (!res.ok) throw new Error("couldn't make a random position");
  return (await res.json()).fen;
}

export function squareName(file: number, rank: number): string {
  return FILES[file] + (rank + 1);
}

export function squareCoords(sq: string): [number, number] {
  return [FILES.indexOf(sq[0]), Number(sq[1]) - 1];
}

/** Placement + side to move -> FEN (no castling or en passant). */
export function toFen(placement: Placement, turn: Color): string {
  const rows: string[] = [];
  for (let rank = 7; rank >= 0; rank--) {
    let row = "";
    let empty = 0;
    for (let file = 0; file < 8; file++) {
      const p = placement[squareName(file, rank)];
      if (!p) {
        empty++;
        continue;
      }
      if (empty) row += empty;
      empty = 0;
      row += p[0] === "w" ? p[1] : p[1].toLowerCase();
    }
    rows.push(row + (empty ? empty : ""));
  }
  return `${rows.join("/")} ${turn} - - 0 1`;
}

/** FEN -> placement + side to move. */
export function fromFen(fen: string): { placement: Placement; turn: Color } {
  const [board, turn] = fen.split(" ");
  const placement: Placement = {};
  board.split("/").forEach((row, i) => {
    const rank = 7 - i;
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) {
        file += Number(ch);
      } else {
        const color: Color = ch === ch.toUpperCase() ? "w" : "b";
        placement[squareName(file, rank)] = `${color}${ch.toUpperCase()}` as PieceCode;
        file++;
      }
    }
  });
  return { placement, turn: (turn as Color) ?? "w" };
}

export function pct(x: number): string {
  return `${Math.round(x * 100)}%`;
}
