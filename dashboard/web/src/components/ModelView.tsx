// The "Model" tab: a picture of the network (layers, sizes, weights), what it
// sees for the current position (its 769 inputs as 12 mini boards plus the
// side to move), its output for that position, and its training record.
import type { Analysis, ModelInfo } from "../lib";
import { pct } from "../lib";

const PLANES = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"];
const OUT_LABELS = ["Loss", "Draw", "Win"];

function fmt(n: number) {
  return n.toLocaleString("en-US");
}

function Section({ title, children, right }: { title: string; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-white/5 bg-ink-900/80 p-5 shadow-lg shadow-black/20">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h3 className="text-[11px] font-semibold uppercase tracking-[0.14em] text-ink-300">{title}</h3>
        {right}
      </div>
      {children}
    </section>
  );
}

/** Layer diagram: columns of nodes joined by faint connections. */
function Diagram({ info, output }: { info: ModelInfo; output: number[] | null }) {
  const W = 960;
  const H = 360;
  const n = info.sizes.length;
  const xs = info.sizes.map((_, i) => 90 + (i * (W - 180)) / (n - 1));
  const shown = (size: number) => Math.min(size, size > 12 ? 9 : size);
  const heights = info.sizes.map((s) => Math.max(70, Math.min(260, 40 + Math.log2(s) * 24)));
  const nodeY = (i: number) => {
    const k = shown(info.sizes[i]);
    const h = heights[i];
    return Array.from({ length: k }, (_, j) => H / 2 - h / 2 + 18 + (j * (h - 36)) / Math.max(1, k - 1));
  };
  const ys = info.sizes.map((_, i) => nodeY(i));
  const titles = info.sizes.map((_, i) =>
    i === 0 ? "Input" : i === n - 1 ? "Output" : `Hidden ${i}`,
  );
  const subtitle = (i: number) =>
    i === 0 ? "board + turn" : i === n - 1 ? "softmax" : "ReLU";

  return (
    <svg viewBox={`0 0 ${W} ${H + 70}`} className="w-full">
      <defs>
        <linearGradient id="edge" x1="0" x2="1">
          <stop offset="0%" stopColor="#6ee7b7" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#818cf8" stopOpacity="0.35" />
        </linearGradient>
      </defs>
      {/* Connections. */}
      {xs.slice(0, -1).map((x, i) =>
        ys[i].map((y1, a) =>
          ys[i + 1].map((y2, b) => (
            <line key={`${i}-${a}-${b}`} x1={x + 9} y1={y1} x2={xs[i + 1] - 9} y2={y2} stroke="url(#edge)" strokeWidth="0.8" />
          )),
        ),
      )}
      {/* Weight counts between layers. */}
      {info.layers.map((l, i) => (
        <g key={`w${i}`}>
          <rect x={(xs[i] + xs[i + 1]) / 2 - 62} y={H - 16} width="124" height="24" rx="12" className="fill-ink-800" />
          <text x={(xs[i] + xs[i + 1]) / 2} y={H} textAnchor="middle" className="fill-ink-300 text-[12px]">
            {fmt(l.params)} weights
          </text>
        </g>
      ))}
      {/* Layers. */}
      {xs.map((x, i) => {
        const h = heights[i];
        const isOut = i === n - 1;
        return (
          <g key={`l${i}`}>
            <rect x={x - 26} y={H / 2 - h / 2} width="52" height={h} rx="26" className="fill-ink-850 stroke-white/10" />
            {ys[i].map((y, j) => {
              const p = isOut && output ? output[j] : null;
              return (
                <g key={j}>
                  <circle cx={x} cy={y} r={isOut ? 9 : 6}
                    className={isOut ? "fill-emerald-300" : i === 0 ? "fill-zinc-200" : "fill-indigo-300"}
                    fillOpacity={p === null ? 0.9 : 0.25 + 0.75 * p} />
                  {isOut && (
                    <text x={x + 18} y={y + 4} className="fill-zinc-200 text-[13px] font-medium">
                      {OUT_LABELS[j]}{p !== null ? ` ${pct(p)}` : ""}
                    </text>
                  )}
                </g>
              );
            })}
            {shown(info.sizes[i]) < info.sizes[i] && (
              <text x={x} y={H / 2 + h / 2 + 14} textAnchor="middle" className="fill-ink-500 text-[12px]">⋮</text>
            )}
            <text x={x} y={H + 36} textAnchor="middle" className="fill-zinc-100 text-[14px] font-semibold">{titles[i]}</text>
            <text x={x} y={H + 54} textAnchor="middle" className="fill-ink-300 text-[12px]">
              {fmt(info.sizes[i])} · {subtitle(i)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function MiniBoard({ active, label }: { active: Set<number>; label: string }) {
  const cells = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const sq = (7 - row) * 8 + col;
      const on = active.has(sq);
      cells.push(
        <div key={sq} className={on ? "bg-emerald-300 shadow-[0_0_8px_rgba(110,231,183,0.8)]" : (row + col) % 2 ? "bg-ink-700/70" : "bg-ink-800"} />,
      );
    }
  }
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="grid aspect-square w-full grid-cols-8 overflow-hidden rounded-md ring-1 ring-white/5">{cells}</div>
      <span className="text-[11px] text-ink-300">{label}</span>
    </div>
  );
}

function Inputs({ input }: { input: NonNullable<ModelInfo["input"]> }) {
  const byPlane = Array.from({ length: 12 }, () => new Set<number>());
  for (const idx of input.active) if (idx < 768) byPlane[Math.floor(idx / 64)].add(idx % 64);
  const whiteToMove = input.active.includes(768);
  return (
    <div className="flex flex-col gap-4">
      {[0, 1].map((color) => (
        <div key={color} className="grid grid-cols-6 gap-3">
          {PLANES.map((name, t) => (
            <MiniBoard key={name} active={byPlane[color * 6 + t]} label={`${color ? "Black" : "White"} ${name.toLowerCase()}`} />
          ))}
        </div>
      ))}
      <div className="flex flex-wrap items-center gap-3 text-sm text-ink-300">
        <span className={`rounded-md px-2 py-1 font-mono text-xs ${whiteToMove ? "bg-emerald-300 text-ink-950" : "bg-ink-800"}`}>
          input 769 = {whiteToMove ? 1 : 0}
        </span>
        <span>side to move ({whiteToMove ? "White" : "Black"})</span>
        <span className="text-ink-500">·</span>
        <span>{input.active.length} of 769 inputs are 1, the rest are 0</span>
      </div>
      {input.flipped && (
        <p className="rounded-lg bg-indigo-400/10 px-3 py-2 text-xs text-indigo-200 ring-1 ring-indigo-400/20">
          Black has more material here, so the network sees the position with colours swapped and the board mirrored, with
          the stronger side as White, exactly as in training. Its answer is converted back afterwards.
        </p>
      )}
    </div>
  );
}

function Training({ info }: { info: ModelInfo }) {
  const t = info.training;
  const r = info.results;
  if (!t && !r) {
    return (
      <p className="text-sm text-ink-300">
        No training record found. Put <code className="rounded bg-ink-800 px-1">training_log.json</code> and{" "}
        <code className="rounded bg-ink-800 px-1">results.json</code> from the Colab download next to{" "}
        <code className="rounded bg-ink-800 px-1">model/warpmate_net.pt</code>.
      </p>
    );
  }
  const accs = t?.epochs.map((e) => e.test_accuracy) ?? [];
  const lo = Math.min(...accs, t?.baseline_accuracy ?? 1);
  const stats: [string, string][] = [];
  if (t?.epochs.length) stats.push(["Test accuracy", pct(t.epochs[t.epochs.length - 1].test_accuracy)]);
  if (t) stats.push(["Baseline", pct(t.baseline_accuracy)], ["Training positions", fmt(t.train_examples)], ["Test positions", fmt(t.test_examples)]);
  if (r?.moves) stats.push(["Keeps the best result", pct(r.moves.overall)]);
  return (
    <div className="flex flex-col gap-5">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {stats.map(([k, v]) => (
          <div key={k} className="rounded-xl bg-ink-850 p-3 ring-1 ring-white/5">
            <div className="text-lg font-semibold tabular-nums">{v}</div>
            <div className="text-[11px] text-ink-300">{k}</div>
          </div>
        ))}
      </div>
      {accs.length > 0 && (
        <div>
          <div className="mb-2 text-xs text-ink-300">Test accuracy after each pass over the training data</div>
          <div className="flex h-28 items-end gap-1.5">
            {accs.map((a, i) => (
              <div key={i} className="flex flex-1 flex-col items-center gap-1">
                <span className="text-[10px] tabular-nums text-ink-300">{(a * 100).toFixed(1)}</span>
                <div className="w-full rounded-t-md bg-gradient-to-t from-emerald-400/40 to-emerald-300"
                  style={{ height: `${8 + ((a - lo) / Math.max(1e-6, 1 - lo)) * 80}px` }} />
                <span className="text-[10px] text-ink-500">{i + 1}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      {r?.five_piece && r.five_piece.length > 0 && (
        <div>
          <div className="mb-2 text-xs text-ink-300">5-piece endgames it never saw (Syzygy answers)</div>
          <div className="flex flex-wrap gap-2">
            {r.five_piece.map((f) => (
              <span key={f.endgame} className="rounded-lg bg-ink-850 px-2.5 py-1.5 text-xs ring-1 ring-white/5">
                <span className="font-mono font-semibold">{f.endgame}</span>{" "}
                <span className="text-emerald-300">{pct(f.accuracy)}</span>{" "}
                <span className="text-ink-500">vs {pct(f.baseline)} baseline</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export function ModelView({ info, analysis, error }: { info: ModelInfo | null; analysis: Analysis | null; error: string | null }) {
  if (error) return <Section title="Model"><p className="text-sm text-rose-300">{error}</p></Section>;
  if (!info) return <Section title="Model"><p className="text-sm text-ink-500">Loading…</p></Section>;

  // Output for the current position, for the side to move: loss, draw, win.
  let output: number[] | null = null;
  const net = analysis?.valid ? analysis.network : undefined;
  if (net && !net.placeholder) {
    output = analysis!.turn === "w" ? [net.black, net.draw, net.white] : [net.white, net.draw, net.black];
  }

  return (
    <div className="flex flex-col gap-6">
      <Section
        title="Architecture"
        right={
          <div className="flex gap-2 text-[11px]">
            <span className={`rounded-full px-2 py-0.5 ${info.loaded ? "bg-emerald-400/15 text-emerald-300" : "bg-amber-400/15 text-amber-300"}`}>
              {info.loaded ? "trained weights loaded" : "no trained weights yet"}
            </span>
            <span className="rounded-full bg-ink-700/70 px-2 py-0.5 text-ink-300">{fmt(info.total_params)} weights</span>
          </div>
        }
      >
        <Diagram info={info} output={output} />
        <div className="mt-2 grid gap-3 text-sm text-ink-300 sm:grid-cols-3">
          <p><span className="font-semibold text-zinc-100">1. Read the board.</span> The position becomes 769 numbers: a 1 for each piece on its square, and one for whose turn it is.</p>
          <p><span className="font-semibold text-zinc-100">2. Mix.</span> Each layer multiplies by its weights and adds them up; ReLU turns negatives into 0.</p>
          <p><span className="font-semibold text-zinc-100">3. Decide.</span> The last 3 numbers go through softmax, giving loss / draw / win percentages for the side to move.</p>
        </div>
      </Section>

      <Section title="What the network sees for this position">
        {info.input ? <Inputs input={info.input} /> : <p className="text-sm text-ink-500">Set up a legal position on the board tab.</p>}
      </Section>

      <Section title="Training">
        <Training info={info} />
      </Section>
    </div>
  );
}
