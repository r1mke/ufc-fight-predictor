"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, RefreshCw, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { PendingFightForm } from "@/components/PendingFightForm";
import {
  deletePendingFight,
  getModelVersions,
  getPendingFights,
  getRetrainStatus,
  getScrapeStatus,
  triggerRetrain,
  triggerScrape,
} from "@/lib/api";
import type { JobStatus, ModelVersion, PendingFight } from "@/types";

// Admin page is free access for now - the backend doesn't check this value,
// it's only kept because the API functions take a token argument.
const ADMIN_TOKEN = "free-access";

function StatusBadge({ status }: { status: PendingFight["status"] }) {
  const variant = status === "pending" ? "secondary" : status === "approved" ? "default" : "outline";
  return <Badge variant={variant}>{status}</Badge>;
}

function formatElapsed(startedAt: string | null | undefined, now: number): string {
  if (!startedAt) return "";
  const started = new Date(startedAt).getTime();
  const totalSeconds = Math.max(0, Math.floor((now - started) / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function isAlreadyRunningError(err: unknown): boolean {
  return err instanceof Error && err.message.toLowerCase().includes("already running");
}

export default function AdminPage() {
  const [pending, setPending] = useState<PendingFight[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [editing, setEditing] = useState<PendingFight | "new" | null>(null);

  const [scrapeStatus, setScrapeStatus] = useState<JobStatus | null>(null);
  const [retrainStatus, setRetrainStatus] = useState<JobStatus | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  const scrapePollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retrainPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Tick once a second while either job is running, to drive the elapsed-time display.
  useEffect(() => {
    if (scrapeStatus?.status !== "running" && retrainStatus?.status !== "running") return;
    const interval = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(interval);
  }, [scrapeStatus?.status, retrainStatus?.status]);

  async function refreshAll(token: string) {
    try {
      const [pendingList, versionList] = await Promise.all([
        getPendingFights(token),
        getModelVersions(token),
      ]);
      setPending(pendingList);
      setVersions(versionList.versions);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load admin data");
    }
  }

  function pollUntilDone(
    ref: React.RefObject<ReturnType<typeof setInterval> | null>,
    getStatus: () => Promise<JobStatus>,
    onUpdate: (s: JobStatus) => void,
    onDone: () => void
  ) {
    if (ref.current) clearInterval(ref.current);
    ref.current = setInterval(async () => {
      const status = await getStatus();
      onUpdate(status);
      if (status.status === "done" || status.status === "error") {
        clearInterval(ref.current!);
        onDone();
      }
    }, 2000);
  }

  useEffect(() => {
    refreshAll(ADMIN_TOKEN);
    // A scrape/retrain may already be running (started from another tab, or still
    // finishing from before a page reload) - sync onto it instead of losing track.
    (async () => {
      const [scrape, retrain] = await Promise.all([
        getScrapeStatus(ADMIN_TOKEN),
        getRetrainStatus(ADMIN_TOKEN),
      ]);
      setScrapeStatus(scrape);
      setRetrainStatus(retrain);
      if (scrape.status === "running") {
        pollUntilDone(scrapePollRef, () => getScrapeStatus(ADMIN_TOKEN), setScrapeStatus, () => refreshAll(ADMIN_TOKEN));
      }
      if (retrain.status === "running") {
        pollUntilDone(retrainPollRef, () => getRetrainStatus(ADMIN_TOKEN), setRetrainStatus, () => {
          setSelected(new Set());
          refreshAll(ADMIN_TOKEN);
        });
      }
    })();
  }, []);

  async function handleScrape() {
    try {
      const status = await triggerScrape(ADMIN_TOKEN);
      setScrapeStatus(status);
      pollUntilDone(scrapePollRef, () => getScrapeStatus(ADMIN_TOKEN), setScrapeStatus, () => refreshAll(ADMIN_TOKEN));
    } catch (err) {
      if (!isAlreadyRunningError(err)) {
        setError(err instanceof Error ? err.message : "Failed to start scrape");
        return;
      }
      // Someone else already kicked one off - just sync onto it instead of erroring.
      const status = await getScrapeStatus(ADMIN_TOKEN);
      setScrapeStatus(status);
      if (status.status === "running") {
        pollUntilDone(scrapePollRef, () => getScrapeStatus(ADMIN_TOKEN), setScrapeStatus, () => refreshAll(ADMIN_TOKEN));
      }
    }
  }

  async function handleRetrain() {
    if (selected.size === 0) return;
    try {
      const status = await triggerRetrain(ADMIN_TOKEN, Array.from(selected));
      setRetrainStatus(status);
      pollUntilDone(retrainPollRef, () => getRetrainStatus(ADMIN_TOKEN), setRetrainStatus, () => {
        setSelected(new Set());
        refreshAll(ADMIN_TOKEN);
      });
    } catch (err) {
      if (!isAlreadyRunningError(err)) {
        setError(err instanceof Error ? err.message : "Failed to start retrain");
        return;
      }
      const status = await getRetrainStatus(ADMIN_TOKEN);
      setRetrainStatus(status);
      if (status.status === "running") {
        pollUntilDone(retrainPollRef, () => getRetrainStatus(ADMIN_TOKEN), setRetrainStatus, () => {
          setSelected(new Set());
          refreshAll(ADMIN_TOKEN);
        });
      }
    }
  }

  async function handleReject(id: string) {
    await deletePendingFight(ADMIN_TOKEN, id);
    refreshAll(ADMIN_TOKEN);
  }

  function toggleSelected(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const pendingOnly = pending.filter((p) => p.status === "pending");

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 space-y-8 px-4 py-10 sm:px-6">
      <header>
        <h1 className="text-2xl font-semibold tracking-tight">Admin</h1>
        <p className="text-sm text-muted-foreground">Fetch new fights, review them, and retrain the models.</p>
      </header>

      {error && (
        <p className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{error}</p>
      )}

      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle>New fights</CardTitle>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setEditing("new")}>Add manually</Button>
            <Button size="sm" onClick={handleScrape} disabled={scrapeStatus?.status === "running"}>
              {scrapeStatus?.status === "running" ? (
                <Loader2 className="mr-2 size-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 size-4" />
              )}
              Check for new fights
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          {scrapeStatus && (
            <p className="text-sm text-muted-foreground">
              {scrapeStatus.status === "running"
                ? `Checking ufcstats.com for new fights... ${formatElapsed(scrapeStatus.started_at, now)} elapsed. This can take a few minutes depending on how many new events there are.`
                : scrapeStatus.status === "error"
                  ? scrapeStatus.error_message
                  : scrapeStatus.result_summary}
            </p>
          )}

          {editing && (
            <PendingFightForm
              adminToken={ADMIN_TOKEN}
              fight={editing === "new" ? undefined : editing}
              onCancel={() => setEditing(null)}
              onSaved={() => {
                setEditing(null);
                refreshAll(ADMIN_TOKEN);
              }}
            />
          )}

          {pendingOnly.length === 0 && !editing && (
            <p className="text-sm text-muted-foreground">Nothing pending. Check for new fights or add one manually.</p>
          )}

          {pendingOnly.map((fight) => (
            <div key={fight.id} className="flex items-center gap-3 rounded-lg border p-3">
              <Checkbox checked={selected.has(fight.id)} onCheckedChange={() => toggleSelected(fight.id)} />
              <button className="flex-1 text-left" onClick={() => setEditing(fight)}>
                <p className="text-sm font-medium">
                  {fight.fighter1_name} vs {fight.fighter2_name}
                </p>
                <p className="text-xs text-muted-foreground">
                  {fight.event_date} · {fight.weight_class} · {fight.method} · winner: {fight.winner_name}
                </p>
              </button>
              <Badge variant="outline">{fight.source}</Badge>
              {(fight.new_fighter_1 || fight.new_fighter_2) && <Badge>new fighter</Badge>}
              <StatusBadge status={fight.status} />
              <Button variant="ghost" size="sm" onClick={() => handleReject(fight.id)}>Reject</Button>
            </div>
          ))}

          <div className="flex items-center justify-between border-t pt-3">
            <p className="text-sm text-muted-foreground">{selected.size} selected</p>
            <Button onClick={handleRetrain} disabled={selected.size === 0 || retrainStatus?.status === "running"}>
              {retrainStatus?.status === "running" && <Loader2 className="mr-2 size-4 animate-spin" />}
              Retrain with selected
            </Button>
          </div>
          {retrainStatus && (
            <p className="text-sm text-muted-foreground">
              {retrainStatus.status === "running"
                ? `Retraining models... ${formatElapsed(retrainStatus.started_at, now)} elapsed.`
                : retrainStatus.status === "error"
                  ? retrainStatus.error_message
                  : retrainStatus.result_summary}
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Model version history</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          {versions.map((v) => (
            <div key={v.version_id} className="flex items-center justify-between rounded-lg border p-3 text-sm">
              <div>
                <p className="font-medium">
                  {v.version_id} {v.is_current && <Badge className="ml-2">current</Badge>}
                </p>
                <p className="text-xs text-muted-foreground">
                  Winner (LightGBM) acc: {v.metrics_summary.winner?.lightgbm?.accuracy?.toFixed(3) ?? "—"} ·
                  {" "}Method (LightGBM) acc: {v.metrics_summary.method?.lightgbm?.accuracy?.toFixed(3) ?? "—"}
                </p>
              </div>
              {!v.is_current && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    const { restoreModelVersion } = await import("@/lib/api");
                    await restoreModelVersion(ADMIN_TOKEN, v.version_id);
                    refreshAll(ADMIN_TOKEN);
                  }}
                >
                  <RotateCcw className="mr-2 size-4" /> Restore
                </Button>
              )}
            </div>
          ))}
          {versions.length === 0 && <p className="text-sm text-muted-foreground">No versions yet.</p>}
        </CardContent>
      </Card>
    </main>
  );
}
