import { useQuery } from '@tanstack/react-query';
import { useEffect } from 'react';
import type { UseFormReturn } from 'react-hook-form';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Toggle } from '@/components/ui/toggle';
import { apiClient } from '@/lib/api/client';
import type { ParamSpec } from '@/lib/api/types';
import type { GenerationFormValues } from '@/lib/hooks/useGenerationForm';

interface AdvancedGeneratePanelProps {
  form: UseFormReturn<GenerationFormValues>;
  engine: string;
}

/**
 * Advanced-mode controls that build themselves from the backend capability
 * endpoints (FORK_NOTES §7d): per-engine inference params (GET /engines) and
 * the verify-loop config (GET /verify/params). On a backend that doesn't
 * expose these (e.g. an older server) the queries just return nothing and the
 * panel degrades to the verify toggle only.
 */
export function AdvancedGeneratePanel({ form, engine }: AdvancedGeneratePanelProps) {
  const { data: engines } = useQuery({
    queryKey: ['engines'],
    queryFn: () => apiClient.listEngines(),
    staleTime: 60_000,
    retry: false,
  });

  const verify = form.watch('verify') ?? false;

  const { data: verifyParams } = useQuery({
    queryKey: ['verifyParams'],
    queryFn: () => apiClient.getVerifyParams(),
    enabled: verify,
    staleTime: 60_000,
    retry: false,
  });

  const engineSpec =
    engines?.engines.find((e) => e.engine === engine)?.param_spec.filter((p) => p.stage !== 'load') ??
    [];
  const ttsParams = (form.watch('ttsParams') ?? {}) as Record<string, number>;
  const verifyConfig = (form.watch('verifyConfig') ?? {}) as Record<
    string,
    number | boolean | string
  >;

  // Clear engine-param overrides when the engine changes — they're validated
  // per-engine, so stale keys from another engine would 422 the request.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => form.setValue('ttsParams', {}), [engine]);

  const setTts = (name: string, v: number) =>
    form.setValue('ttsParams', { ...ttsParams, [name]: v }, { shouldDirty: true });
  const setVerifyCfg = (name: string, v: number | boolean | string) =>
    form.setValue('verifyConfig', { ...verifyConfig, [name]: v }, { shouldDirty: true });

  function renderControl(
    spec: ParamSpec,
    value: number | boolean | string | undefined,
    onChange: (v: number | boolean | string) => void,
  ) {
    if (typeof spec.default === 'boolean') {
      return (
        <Toggle
          checked={Boolean(value ?? spec.default)}
          onCheckedChange={(c) => onChange(c)}
        />
      );
    }
    // Numeric slider — either a concrete numeric default, or an OPTIONAL numeric
    // (default null with numeric bounds, e.g. TADA's speed_up_factor). The
    // optional case reads "off" until touched and is only sent once the user
    // moves it, so the engine keeps its own default.
    const hasNumericDefault = typeof spec.default === 'number';
    const isOptionalNumeric =
      spec.default == null && typeof spec.min === 'number' && typeof spec.max === 'number';
    if (hasNumericDefault || isOptionalNumeric) {
      const min = typeof spec.min === 'number' ? spec.min : 0;
      const max = typeof spec.max === 'number' ? spec.max : 1;
      const isInt =
        hasNumericDefault &&
        Number.isInteger(spec.default as number) &&
        Number.isInteger(min) &&
        Number.isInteger(max);
      const step = isInt ? 1 : Math.max(0.01, Number(((max - min) / 100).toFixed(2)));
      const isSet = value !== undefined;
      const cur = Number(value ?? (hasNumericDefault ? (spec.default as number) : min));
      return (
        <div className="flex flex-1 items-center gap-2">
          <Slider
            value={[cur]}
            min={min}
            max={max}
            step={step}
            onValueChange={(v) => onChange(v[0])}
            className="flex-1"
          />
          <span className="w-12 text-right text-xs tabular-nums text-muted-foreground">
            {!isSet && isOptionalNumeric ? 'off' : isInt ? cur : cur.toFixed(2)}
          </span>
        </div>
      );
    }
    return (
      <Input
        value={String(value ?? spec.default)}
        onChange={(e) => onChange(e.target.value)}
        className="h-7 w-28 text-xs"
      />
    );
  }

  return (
    <div className="mt-3 space-y-3 rounded-2xl border border-border bg-card/50 p-3">
      {engineSpec.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground">Engine parameters</div>
          {engineSpec.map((p) => (
            <div key={p.name} className="flex items-center gap-3" title={p.desc}>
              <Label className="w-40 shrink-0 text-xs">{p.name}</Label>
              {renderControl(p, ttsParams[p.name], (v) => setTts(p.name, v as number))}
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs font-medium">Verify loop</div>
          <div className="text-[11px] text-muted-foreground">
            Transcribe each chunk; re-seed then split on mismatch.
          </div>
        </div>
        <Toggle checked={verify} onCheckedChange={(c) => form.setValue('verify', c)} />
      </div>

      {verify && verifyParams && verifyParams.param_spec.length > 0 && (
        <div className="space-y-2 border-t border-border pt-2">
          <div className="text-xs font-medium text-muted-foreground">Verify config</div>
          {verifyParams.param_spec.map((p) => (
            <div key={p.name} className="flex items-center gap-3" title={p.desc}>
              <Label className="w-40 shrink-0 text-xs">{p.name}</Label>
              {renderControl(p, verifyConfig[p.name], (v) => setVerifyCfg(p.name, v))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
