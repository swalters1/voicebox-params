import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  AlertTriangle,
  DatabaseBackup,
  HardDriveDownload,
  Loader2,
  RotateCcw,
  ShieldCheck,
} from 'lucide-react';
import { useState } from 'react';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
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

  const { data: pending } = useQuery({
    queryKey: ['backups', 'pending'],
    queryFn: () => apiClient.getPendingRestore(),
    retry: false,
  });

  const [confirming, setConfirming] = useState<BackupResponse | null>(null);

  const restore = useMutation({
    mutationFn: (name: string) => apiClient.restoreBackup(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['backups'] });
      toast({
        title: 'Restore staged',
        description: 'Restart Voicebox to apply it. Your current database will be saved first.',
      });
    },
    onError: (error: unknown) => {
      toast({
        title: 'Cannot restore this backup',
        description: error instanceof Error ? error.message : 'The backup was rejected',
        variant: 'destructive',
      });
    },
  });

  const cancelRestore = useMutation({
    mutationFn: () => apiClient.cancelPendingRestore(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['backups'] });
      toast({ title: 'Restore cancelled' });
    },
  });

  const backups = data?.backups ?? [];
  const automatic = backups.filter((b) => b.kind === 'automatic').length;
  const isPending = pending?.pending ?? false;

  return (
    <SettingSection
      title="Backups"
      description="Snapshots of the database — voices, history, seeds, and settings. Rendered audio is not included; it can be regenerated from the seeds each record stores."
    >
      {isPending && (
        <SettingRow
          title="Restore pending"
          description="A restore is staged. Restart Voicebox to apply it — the current database will be saved aside first."
          action={
            <Button
              onClick={() => cancelRestore.mutate()}
              disabled={cancelRestore.isPending}
              variant="outline"
              size="sm"
            >
              Cancel
            </Button>
          }
        >
          <div className="flex items-center gap-2 rounded-lg border border-accent/30 bg-accent/5 px-3 py-2 text-sm">
            <AlertTriangle className="h-4 w-4 shrink-0 text-accent" />
            <span className="text-muted-foreground">
              Nothing has changed yet. The swap happens on next start.
            </span>
          </div>
        </SettingRow>
      )}

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
                <Button
                  onClick={() => setConfirming(b)}
                  disabled={isPending || restore.isPending}
                  variant="ghost"
                  size="sm"
                  className="shrink-0 h-7 px-2 text-xs"
                >
                  <RotateCcw className="h-3 w-3 mr-1" />
                  Restore
                </Button>
              </div>
            ))}
          </div>
        )}
      </SettingRow>

      <AlertDialog open={confirming !== null} onOpenChange={(open) => !open && setConfirming(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Restore this backup?</AlertDialogTitle>
            <AlertDialogDescription asChild>
              <div className="space-y-2">
                <p>
                  This replaces your voices, history and settings with the contents of{' '}
                  <span className="font-medium text-foreground">
                    v{confirming?.version} · {confirming && formatWhen(confirming.created_at)}
                  </span>
                  .
                </p>
                <p>
                  Nothing changes until you restart. Your current database is saved aside first, so
                  this can be undone.
                </p>
              </div>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                if (confirming) restore.mutate(confirming.name);
                setConfirming(null);
              }}
            >
              Stage restore
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </SettingSection>
  );
}
