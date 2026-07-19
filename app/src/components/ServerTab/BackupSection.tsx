import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { DatabaseBackup, HardDriveDownload, Loader2, ShieldCheck } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/components/ui/use-toast';
import { apiClient } from '@/lib/api/client';
import type { BackupResponse } from '@/lib/api/types';
import { SettingRow, SettingSection } from './SettingRow';

function formatSize(bytes: number): string {
  if (bytes >= 1_000_000_000) return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
  if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1000))} KB`;
}

function formatWhen(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleString();
}

/**
 * Database backups.
 *
 * Two kinds land in the same directory and the same list:
 *  - `automatic` — taken by the server immediately before a schema migration,
 *    one per app version. Covers the upgrade, which is the moment the data is
 *    actually at risk and the moment nobody is thinking about backups.
 *  - `manual` — this button. Runs against a live database via VACUUM INTO, so
 *    it is safe to press while renders are in flight.
 */
export function BackupSection() {
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ['backups'],
    queryFn: () => apiClient.listBackups(),
    retry: false,
  });

  const backupNow = useMutation({
    mutationFn: () => apiClient.createBackup(),
    onSuccess: (backup: BackupResponse) => {
      queryClient.invalidateQueries({ queryKey: ['backups'] });
      toast({
        title: 'Backup created',
        description: `${backup.name} (${formatSize(backup.size_bytes)})`,
      });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Backup failed',
        description: error instanceof Error ? error.message : 'Could not back up the database',
        variant: 'destructive',
      });
    },
  });

  const backups = data?.backups ?? [];
  const automatic = backups.filter((b) => b.kind === 'automatic').length;

  return (
    <SettingSection
      title="Backups"
      description="Snapshots of the database — voices, history, seeds, and settings. Rendered audio is not included; it can be regenerated from the seeds each record stores."
    >
      <SettingRow
        title="Back up now"
        description="Takes a consistent snapshot even while the server is busy."
        action={
          <Button
            onClick={() => backupNow.mutate()}
            disabled={backupNow.isPending}
            variant="outline"
            size="sm"
          >
            {backupNow.isPending ? (
              <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
            ) : (
              <HardDriveDownload className="h-3.5 w-3.5 mr-1.5" />
            )}
            Back up now
          </Button>
        }
      />

      <SettingRow
        title="Before every upgrade"
        description={
          automatic > 0
            ? `The database is snapshotted automatically before each migration. ${automatic} kept.`
            : 'The database is snapshotted automatically before each migration.'
        }
        action={<ShieldCheck className="h-4 w-4 text-accent" />}
      />

      <SettingRow title="Saved backups" description={data?.directory}>
        {isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading…
          </div>
        ) : backups.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No backups yet. One is taken automatically the next time an upgrade migrates the
            database.
          </p>
        ) : (
          <div className="divide-y divide-border/40 rounded-lg border border-border/60">
            {backups.map((b) => (
              <div key={b.name} className="flex items-center gap-3 px-3 py-2">
                <DatabaseBackup className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs font-medium">v{b.version}</div>
                  <div className="truncate text-[11px] text-muted-foreground">
                    {formatWhen(b.created_at)} · {formatSize(b.size_bytes)}
                  </div>
                </div>
                <span
                  className={`shrink-0 text-[10px] px-1.5 py-0.5 rounded-full ${
                    b.kind === 'manual'
                      ? 'bg-accent/15 text-accent'
                      : 'bg-muted text-muted-foreground'
                  }`}
                >
                  {b.kind === 'manual' ? 'manual' : 'pre-upgrade'}
                </span>
              </div>
            ))}
          </div>
        )}
      </SettingRow>
    </SettingSection>
  );
}
