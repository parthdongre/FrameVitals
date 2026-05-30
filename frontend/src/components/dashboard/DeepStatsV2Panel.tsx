import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface NumericColumnStats {
  count?: number;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  skewness?: number | null;
  skewness_label?: string;
  kurtosis?: number | null;
  kurtosis_label?: string;
  outliers?: { iqr: number; z3: number; mad_z: number; n: number };
  normality?: {
    shapiro?: { p_value?: number };
    dagostino?: { p_value?: number };
    is_probably_normal?: boolean | null;
    interpretation?: string;
  };
  distribution_fit?: {
    available?: boolean;
    best_fit?: { name: string; aic: number };
  };
  bootstrap_mean_ci?: { available?: boolean; low?: number; high?: number; method?: string };
}

interface NumericPair {
  column_a: string;
  column_b: string;
  pearson?: { r: number | null; p: number | null; strength?: string };
  spearman?: { rho: number | null };
  n: number;
}

interface CategoricalPair {
  column_a: string;
  column_b: string;
  cramers_v: number | null;
  p: number | null;
  strength?: string;
}

interface GroupTest {
  group_column: string;
  numeric_column: string;
  test: string;
  p_value: number | null;
  significant: boolean;
}

interface DeepStatsV2PanelProps {
  deepStats?: {
    version?: string;
    summary?: {
      numeric_count: number;
      categorical_count: number;
      numeric_pairs_tested: number;
      categorical_pairs_tested: number;
      binary_numeric_pairs_tested: number;
      group_difference_tests_run: number;
    };
    numeric_statistics?: Record<string, NumericColumnStats>;
    bivariate?: {
      numeric_pairs?: NumericPair[];
      categorical_pairs?: CategoricalPair[];
      group_difference_tests?: GroupTest[];
    };
  };
}

function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    return v.toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  return String(v);
}

export function DeepStatsV2Panel({ deepStats }: DeepStatsV2PanelProps) {
  if (!deepStats || !deepStats.summary) return null;

  const numericStats = deepStats.numeric_statistics ?? {};
  const numericEntries = Object.entries(numericStats).slice(0, 6);
  const numericPairs = (deepStats.bivariate?.numeric_pairs ?? []).slice(0, 6);
  const catPairs = (deepStats.bivariate?.categorical_pairs ?? []).slice(0, 5);
  const groupTests = (deepStats.bivariate?.group_difference_tests ?? []).slice(0, 5);
  const summary = deepStats.summary;

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="overflow-hidden border-white/5 bg-white/[0.03] shadow-panel">
        <CardHeader className="border-b border-white/5 bg-white/[0.015]">
          <CardTitle className="flex items-center gap-3 text-lg text-slate-50">
            <span className="text-cyan-300">Deep statistics</span>
            <span className="text-xs font-mono tracking-[0.3em] text-slate-500">
              v{deepStats.version ?? "2"}
            </span>
          </CardTitle>
          <CardDescription className="text-slate-400">
            Per-column statistics with normality battery, distribution fits, bootstrap CIs and
            bivariate tests.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6 p-6">
          <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
            <Stat label="Numeric cols" value={summary.numeric_count} />
            <Stat label="Categorical cols" value={summary.categorical_count} />
            <Stat label="Numeric pairs" value={summary.numeric_pairs_tested} />
            <Stat label="Cat. pairs" value={summary.categorical_pairs_tested} />
            <Stat label="Bin↔Num" value={summary.binary_numeric_pairs_tested} />
            <Stat label="Group tests" value={summary.group_difference_tests_run} />
          </div>

          {numericEntries.length > 0 ? (
            <section>
              <p className="mb-2 text-xs uppercase tracking-[0.28em] text-slate-500">
                Numeric column highlights (top 6)
              </p>
              <div className="overflow-x-auto rounded-2xl border border-white/5 bg-white/[0.02]">
                <table className="w-full border-separate border-spacing-0 text-left text-sm">
                  <thead className="bg-white/[0.02] text-[11px] uppercase tracking-[0.3em] text-slate-500">
                    <tr>
                      <th className="border-b border-white/5 px-3 py-2">Column</th>
                      <th className="border-b border-white/5 px-3 py-2">Mean</th>
                      <th className="border-b border-white/5 px-3 py-2">Std</th>
                      <th className="border-b border-white/5 px-3 py-2">Skew</th>
                      <th className="border-b border-white/5 px-3 py-2">Kurt</th>
                      <th className="border-b border-white/5 px-3 py-2">Normal?</th>
                      <th className="border-b border-white/5 px-3 py-2">Best-fit</th>
                      <th className="border-b border-white/5 px-3 py-2">Outliers (IQR/Z/MAD)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {numericEntries.map(([col, stats]) => {
                      const fit = stats.distribution_fit?.best_fit;
                      const isNormal = stats.normality?.is_probably_normal;
                      const o = stats.outliers ?? { iqr: 0, z3: 0, mad_z: 0, n: 0 };
                      return (
                        <tr
                          key={col}
                          className="border-b border-white/5 odd:bg-white/[0.01] even:bg-transparent"
                        >
                          <td className="border-b border-white/5 px-3 py-2 text-slate-100">{col}</td>
                          <td className="border-b border-white/5 px-3 py-2 font-mono text-slate-300">
                            {fmt(stats.mean)}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 font-mono text-slate-300">
                            {fmt(stats.std)}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 font-mono text-slate-300">
                            {fmt(stats.skewness)}{" "}
                            <span className="text-[10px] text-slate-500">
                              {stats.skewness_label?.split(" ")[0]}
                            </span>
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 font-mono text-slate-300">
                            {fmt(stats.kurtosis)}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2">
                            {isNormal === null || isNormal === undefined ? (
                              <span className="text-slate-500">—</span>
                            ) : isNormal ? (
                              <Badge variant="muted">yes</Badge>
                            ) : (
                              <Badge variant="violet">no</Badge>
                            )}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 text-slate-300">
                            {fit ? (
                              <span>
                                <span className="font-mono text-cyan-200">{fit.name}</span>{" "}
                                <span className="text-[10px] text-slate-500">
                                  AIC {fmt(fit.aic, 1)}
                                </span>
                              </span>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td className="border-b border-white/5 px-3 py-2 font-mono text-xs text-slate-400">
                            {o.iqr}/{o.z3}/{o.mad_z}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

          <div className="grid gap-4 xl:grid-cols-3">
            <section className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
              <p className="text-xs uppercase tracking-[0.28em] text-slate-500">Top numeric pairs</p>
              {numericPairs.length > 0 ? (
                <ul className="mt-3 space-y-2 text-sm">
                  {numericPairs.map((p) => (
                    <li
                      key={`${p.column_a}-${p.column_b}`}
                      className="flex items-center justify-between gap-2"
                    >
                      <span className="truncate text-slate-200">
                        {p.column_a} ↔ {p.column_b}
                      </span>
                      <span className="font-mono text-cyan-200">{fmt(p.pearson?.r)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-500">No numeric pairs.</p>
              )}
            </section>

            <section className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
              <p className="text-xs uppercase tracking-[0.28em] text-slate-500">
                Cat ↔ Cat (Cramér's V)
              </p>
              {catPairs.length > 0 ? (
                <ul className="mt-3 space-y-2 text-sm">
                  {catPairs.map((p) => (
                    <li
                      key={`${p.column_a}-${p.column_b}`}
                      className="flex items-center justify-between gap-2"
                    >
                      <span className="truncate text-slate-200">
                        {p.column_a} ↔ {p.column_b}
                      </span>
                      <span className="font-mono text-violet-200">{fmt(p.cramers_v)}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-500">No categorical pairs.</p>
              )}
            </section>

            <section className="rounded-2xl border border-white/5 bg-white/[0.02] p-4">
              <p className="text-xs uppercase tracking-[0.28em] text-slate-500">
                Group difference tests
              </p>
              {groupTests.length > 0 ? (
                <ul className="mt-3 space-y-2 text-sm">
                  {groupTests.map((t, i) => (
                    <li key={i} className="flex items-center justify-between gap-2">
                      <span className="truncate text-slate-200">
                        {t.numeric_column} ~ {t.group_column}
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs text-slate-400">{t.test.split(" ")[0]}</span>
                        <span
                          className={`font-mono ${
                            t.significant ? "text-emerald-300" : "text-slate-500"
                          }`}
                        >
                          p={fmt(t.p_value, 4)}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="mt-2 text-sm text-slate-500">No group tests run.</p>
              )}
            </section>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] px-3 py-2">
      <p className="text-[10px] uppercase tracking-[0.28em] text-slate-500">{label}</p>
      <p className="mt-1 font-mono text-lg text-slate-50">{value}</p>
    </div>
  );
}
