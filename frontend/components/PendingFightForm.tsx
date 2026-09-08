"use client";

import { Fragment, useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FighterSearch } from "@/components/FighterSearch";
import { createPendingFight, updatePendingFight } from "@/lib/api";
import { WEIGHT_CLASSES } from "@/lib/constants";
import type { FighterSummary, FightMethod, PendingFight, PendingFightCreate } from "@/types";

const METHODS: { value: FightMethod; label: string }[] = [
  { value: "KO_TKO", label: "KO / TKO" },
  { value: "Submission", label: "Submission" },
  { value: "Decision", label: "Decision" },
];

const STAT_PAIRS: { key: string; label: string }[] = [
  { key: "sig_landed", label: "Significant strikes landed" },
  { key: "sig_att", label: "Significant strikes attempted" },
  { key: "td_landed", label: "Takedowns landed" },
  { key: "td_att", label: "Takedowns attempted" },
  { key: "ctrl_sec", label: "Control time (seconds)" },
  { key: "kd", label: "Knockdowns" },
  { key: "sub_att", label: "Submission attempts" },
];

interface Props {
  adminToken: string;
  fight?: PendingFight;
  onSaved: (fight: PendingFight) => void;
  onCancel: () => void;
}

function emptyForm(): PendingFightCreate {
  return {
    fighter1_id: null,
    fighter2_id: null,
    fighter1_name: "",
    fighter2_name: "",
    weight_class: "",
    method: "Decision",
    winner_name: "",
    event_date: new Date().toISOString().slice(0, 10),
    is_title_fight: false,
    new_fighter_1: null,
    new_fighter_2: null,
    source_url: null,
    f1_sig_landed: null, f1_sig_att: null, f2_sig_landed: null, f2_sig_att: null,
    f1_td_landed: null, f1_td_att: null, f2_td_landed: null, f2_td_att: null,
    f1_ctrl_sec: null, f2_ctrl_sec: null, f1_kd: null, f2_kd: null,
    f1_sub_att: null, f2_sub_att: null,
  };
}

export function PendingFightForm({ adminToken, fight, onSaved, onCancel }: Props) {
  const isEdit = !!fight;
  const [form, setForm] = useState<PendingFightCreate>(fight ?? emptyForm());
  const [fighter1, setFighter1] = useState<FighterSummary | null>(null);
  const [fighter2, setFighter2] = useState<FighterSummary | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function set<K extends keyof PendingFightCreate>(key: K, value: PendingFightCreate[K]) {
    setForm((f) => ({ ...f, [key]: value }));
  }

  function setStat(prefix: "f1" | "f2", key: string, raw: string) {
    const value = raw === "" ? null : Number(raw);
    set(`${prefix}_${key}` as keyof PendingFightCreate, value as never);
  }

  const canSubmit =
    form.fighter1_name && form.fighter2_name && form.weight_class && form.method && form.winner_name && form.event_date;

  async function handleSubmit() {
    setSaving(true);
    setError(null);
    try {
      const saved = isEdit
        ? await updatePendingFight(adminToken, fight!.id, form)
        : await createPendingFight(adminToken, form);
      onSaved(saved);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-4 rounded-xl border bg-card p-5">
      {isEdit ? (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium text-muted-foreground">Fighter 1</label>
            <p className="mt-1.5 text-sm">
              {form.fighter1_name} {form.new_fighter_1 && <span className="text-xs text-primary">(new fighter)</span>}
            </p>
          </div>
          <div>
            <label className="text-sm font-medium text-muted-foreground">Fighter 2</label>
            <p className="mt-1.5 text-sm">
              {form.fighter2_name} {form.new_fighter_2 && <span className="text-xs text-primary">(new fighter)</span>}
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          <FighterSearch
            label="Fighter 1"
            selected={fighter1}
            onSelect={(f) => {
              setFighter1(f);
              set("fighter1_id", f.id);
              set("fighter1_name", f.name);
            }}
            exclude={fighter2?.id}
          />
          <FighterSearch
            label="Fighter 2"
            selected={fighter2}
            onSelect={(f) => {
              setFighter2(f);
              set("fighter2_id", f.id);
              set("fighter2_name", f.name);
            }}
            exclude={fighter1?.id}
          />
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-muted-foreground">Weight class</label>
          <Select value={form.weight_class} onValueChange={(v) => set("weight_class", v)}>
            <SelectTrigger className="w-full"><SelectValue placeholder="Select" /></SelectTrigger>
            <SelectContent>
              {WEIGHT_CLASSES.map((wc) => <SelectItem key={wc} value={wc}>{wc}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-muted-foreground">Method</label>
          <Select value={form.method} onValueChange={(v) => set("method", v as FightMethod)}>
            <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
            <SelectContent>
              {METHODS.map((m) => <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>)}
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-muted-foreground">Winner</label>
          <Select value={form.winner_name} onValueChange={(v) => set("winner_name", v)}>
            <SelectTrigger className="w-full"><SelectValue placeholder="Select" /></SelectTrigger>
            <SelectContent>
              {[form.fighter1_name, form.fighter2_name].filter(Boolean).map((name) => (
                <SelectItem key={name} value={name}>{name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex flex-col gap-1.5">
          <label className="text-sm font-medium text-muted-foreground">Event date</label>
          <Input
            type="date"
            max={new Date().toISOString().slice(0, 10)}
            value={form.event_date?.slice(0, 10) ?? ""}
            onChange={(e) => set("event_date", e.target.value)}
          />
        </div>
      </div>

      <label className="flex items-center gap-2 text-sm">
        <Checkbox checked={form.is_title_fight} onCheckedChange={(v) => set("is_title_fight", !!v)} />
        Title fight
      </label>

      <Accordion type="single" collapsible>
        <AccordionItem value="stats">
          <AccordionTrigger>Detailed statistics (optional)</AccordionTrigger>
          <AccordionContent>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <div />
              <div className="font-medium">{form.fighter1_name || "Fighter 1"}</div>
              <div className="font-medium">{form.fighter2_name || "Fighter 2"}</div>
              {STAT_PAIRS.map(({ key, label }) => (
                <Fragment key={key}>
                  <div className="self-center text-muted-foreground">{label}</div>
                  <Input
                    type="number"
                    value={(form as never as Record<string, number | null>)[`f1_${key}`] ?? ""}
                    onChange={(e) => setStat("f1", key, e.target.value)}
                  />
                  <Input
                    type="number"
                    value={(form as never as Record<string, number | null>)[`f2_${key}`] ?? ""}
                    onChange={(e) => setStat("f2", key, e.target.value)}
                  />
                </Fragment>
              ))}
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={onCancel} disabled={saving}>Cancel</Button>
        <Button onClick={handleSubmit} disabled={!canSubmit || saving}>
          {saving && <Loader2 className="mr-2 size-4 animate-spin" />}
          Save
        </Button>
      </div>
    </div>
  );
}
