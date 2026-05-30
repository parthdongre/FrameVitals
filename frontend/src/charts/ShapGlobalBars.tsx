import { FeatureImportanceBars } from "./FeatureImportanceBars";

interface ShapGlobalBarsProps {
  /**
   * Either the full explainability payload or the bare `global_importance` array.
   */
  explainability: unknown;
  className?: string;
}

/**
 * SHAP global importance — built on top of `FeatureImportanceBars` since the
 * payload shape and rendering are identical. Wrapper exists so MlLab and SHAP
 * tabs each get the right copy.
 */
export function ShapGlobalBars({ explainability, className }: ShapGlobalBarsProps) {
  return (
    <FeatureImportanceBars
      importance={explainability}
      className={className}
      eyebrow="SHAP"
      title="Global SHAP importance"
      description="Mean |SHAP| values across the validation set."
      limit={25}
    />
  );
}
