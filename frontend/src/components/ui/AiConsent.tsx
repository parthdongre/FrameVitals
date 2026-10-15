export function AiConsent({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="flex items-start gap-3 text-left text-xs text-[var(--ink-2)]">
      <input
        className="mt-1"
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>
        Allow this AI request to send dataset information and my question to the
        server’s configured AI provider. This can include an external service.
      </span>
    </label>
  );
}
