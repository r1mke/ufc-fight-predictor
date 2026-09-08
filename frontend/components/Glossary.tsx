import { BookOpen } from "lucide-react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  FEATURE_GLOSSARY,
  FIGHTER_STAT_GLOSSARY,
  MODEL_METRIC_GLOSSARY,
  type GlossaryEntry,
} from "@/lib/glossary";

function GlossaryList({ entries }: { entries: GlossaryEntry[] }) {
  return (
    <dl className="space-y-3">
      {entries.map((entry) => (
        <div key={entry.term}>
          <dt className="text-sm font-medium">{entry.term}</dt>
          <dd className="text-sm text-muted-foreground">{entry.description}</dd>
        </div>
      ))}
    </dl>
  );
}

const FEATURE_ENTRIES: GlossaryEntry[] = Object.values(FEATURE_GLOSSARY);

export function Glossary() {
  return (
    <div className="rounded-xl border bg-card p-5">
      <div className="mb-2 flex items-center gap-2">
        <BookOpen className="size-4 text-muted-foreground" />
        <h2 className="text-sm font-medium">What do these terms mean?</h2>
      </div>
      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="fighter-stats">
          <AccordionTrigger>Fighter stats</AccordionTrigger>
          <AccordionContent>
            <GlossaryList entries={FIGHTER_STAT_GLOSSARY} />
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="model-metrics">
          <AccordionTrigger>Model metrics</AccordionTrigger>
          <AccordionContent>
            <GlossaryList entries={MODEL_METRIC_GLOSSARY} />
          </AccordionContent>
        </AccordionItem>
        <AccordionItem value="prediction-factors">
          <AccordionTrigger>Prediction factors</AccordionTrigger>
          <AccordionContent>
            <p className="mb-3 text-sm text-muted-foreground">
              These are the pre-fight statistics the models compare between the two fighters. Each
              one shown under &ldquo;Most influential factors&rdquo; is the (Fighter A − Fighter B)
              difference in:
            </p>
            <GlossaryList entries={FEATURE_ENTRIES} />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}
