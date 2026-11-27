// Right-hand panel: the network's win/draw/loss guess (always shown), the
// perfect answer from the solved tables (when they cover the position), and
// every legal move ranked.
import type { Analysis, MoveRow } from "../lib";
import { pct } from "../lib";

interface Props {
  analysis: Analysis | null;
  loading: boolean;
  error: string | null;
  canPlayMoves: boolean;
  onPlayMove: (uci: string) => void;
}

function Bar({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-12 text-xs font-medium text-ink-300">{label}</span>
      <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-ink-700/60">
        <div className={`h-full rounded-full transition-[width] duration-500 ease-out ${tone}`} style={{ width: `${value * 100}%` }} />
      </div>
      <span className="w-11 text-right text-sm font-semibold tabular-nums">{pct(value)}</span>
    </div>
  );
}

function Card({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-white/5 bg-ink-900/80 p-4 shadow-lg shadow-black/20 backdrop-blur">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-300">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "warn" | "good" | "bad" }) {
  const tones = {
    neutral: "bg-ink-700/70 text-ink-300",
    warn: "bg-amber-400/15 text-amber-300",
    good: "bg-emerald-400/15 text-emerald-300",
    bad: "bg-rose-400/15 text-rose-300",
  };
  return <span className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${tones[tone]}`}>{children}</span>;
}

function moveResult(m: MoveRow, rankedBy: "perfect" | "network") {
  if (rankedBy === "perfect" && m.perfect) {
    const tone = m.perfect.result > 0 ? "text-emerald-300" : m.perfect.result < 0 ? "text-rose-300" : "text-ink-300";
    return <span className={`text-sm ${tone}`}>{m.perfect.text}</span>;
  }
  return (
    <span className="text-xs tabular-nums text-ink-300">
      win {pct(m.network.win)} · draw {pct(m.network.draw)}
    </span>
  );
}

export function AnalysisPanel({ analysis, loading, error, canPlayMoves, onPlayMove }: Props) {
  if (error) {
    return (
      <Card title="Analysis">
        <p className="text-sm text-rose-300">{error}</p>
        <p className="mt-2 text-xs text-ink-500">Is the analysis server running on port 8000?</p>
      </Card>
    );
  }
  if (!analysis) {
    return <Card title="Analysis"><p className="text-sm text-ink-500">{loading ? "Analyzing…" : "Set up a position to begin."}</p></Card>;
  }
  if (!analysis.valid) {
    return (
      <Card title="Position">
        <p className="text-sm text-amber-300">Not a legal position: {analysis.reason}.</p>
      </Card>
    );
  }

  const net = analysis.network!;
  const perfect = analysis.perfect!;
  const side = analysis.turn === "w" ? "White" : "Black";
  const netPick = net.white >= net.draw && net.white >= net.black ? 1 : net.black >= net.draw ? -1 : 0; // +1 white
  let agrees: boolean | null = null;
  if (perfect.available && !net.placeholder) {
    const truth = analysis.turn === "w" ? perfect.result : -perfect.result; // +1 white wins
    agrees = truth === netPick;
  }
  const moves = analysis.moves ?? [];

  return (
    <div className={`flex flex-col gap-4 transition-opacity ${loading ? "opacity-70" : ""}`}>
      <div className="flex items-baseline justify-between px-1">
        <h2 className="text-lg font-semibold">
          {analysis.game_over ? cap(analysis.game_over) : `${side} to move`}
        </h2>
        <span className="text-xs text-ink-500">{analysis.pieces} pieces</span>
      </div>

      <Card
        title="Network's guess"
        right={
          net.placeholder ? <Badge tone="warn">placeholder: no trained model</Badge>
          : !net.trained_range ? <Badge tone="warn">trained on ≤4 pieces</Badge>
          : <Badge>trained network</Badge>
        }
      >
        <div className="flex flex-col gap-2.5">
          <Bar label="White" value={net.white} tone="bg-zinc-100" />
          <Bar label="Draw" value={net.draw} tone="bg-ink-500" />
          <Bar label="Black" value={net.black} tone="bg-indigo-400" />
        </div>
      </Card>

      <Card
        title="Perfect answer"
        right={agrees === null ? null : agrees ? <Badge tone="good">network agrees</Badge> : <Badge tone="bad">network is wrong</Badge>}
      >
        {perfect.available ? (
          <p className="text-xl font-semibold tracking-tight">{perfect.text}</p>
        ) : (
          <p className="text-sm text-ink-300">No perfect answer: {perfect.why}.</p>
        )}
      </Card>

      {moves.length > 0 && (
        <Card
          title="Best moves"
          right={<span className="text-[11px] text-ink-500">ranked by {analysis.ranked_by === "perfect" ? "perfect play" : "the network"}</span>}
        >
          <ul className="-mx-2 max-h-72 overflow-y-auto pr-1">
            {moves.map((m, i) => (
              <li key={m.uci}>
                <button
                  disabled={!canPlayMoves}
                  onClick={() => onPlayMove(m.uci)}
                  className="flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left transition enabled:hover:bg-white/5 disabled:cursor-default"
                >
                  <span className="w-5 text-right text-xs tabular-nums text-ink-500">{i + 1}</span>
                  <span className="w-16 font-mono text-sm font-semibold">{m.san}</span>
                  <span className="flex-1">{moveResult(m, analysis.ranked_by!)}</span>
                  {m.best && <Badge tone="good">best</Badge>}
                </button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

function cap(s: string) {
  return s[0].toUpperCase() + s.slice(1);
}
