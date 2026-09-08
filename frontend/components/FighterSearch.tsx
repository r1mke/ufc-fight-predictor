"use client";

import { useEffect, useState } from "react";
import { Check, ChevronsUpDown, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { searchFighters } from "@/lib/api";
import type { FighterSummary } from "@/types";

interface Props {
  label: string;
  selected: FighterSummary | null;
  onSelect: (fighter: FighterSummary) => void;
  exclude?: string;
}

export function FighterSearch({ label, selected, onSelect, exclude }: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<FighterSummary[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    const timeout = setTimeout(() => {
      searchFighters(query, 25)
        .then((fighters) => setResults(fighters.filter((f) => f.id !== exclude)))
        .catch(() => setResults([]))
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(timeout);
  }, [query, open, exclude]);

  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm font-medium text-muted-foreground">{label}</label>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between font-normal"
          >
            {selected ? selected.name : "Search fighter..."}
            <ChevronsUpDown className="ml-2 size-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[320px] p-0" align="start">
          <Command shouldFilter={false}>
            <CommandInput
              placeholder="Type a fighter name..."
              value={query}
              onValueChange={setQuery}
            />
            <CommandList>
              {loading && (
                <div className="flex items-center justify-center py-6 text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" />
                </div>
              )}
              {!loading && <CommandEmpty>No fighters found.</CommandEmpty>}
              <CommandGroup>
                {results.map((fighter) => (
                  <CommandItem
                    key={fighter.id}
                    value={fighter.id}
                    onSelect={() => {
                      onSelect(fighter);
                      setOpen(false);
                    }}
                  >
                    <Check
                      className={cn(
                        "mr-2 size-4",
                        selected?.id === fighter.id ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <div className="flex flex-1 flex-col">
                      <span>{fighter.name}</span>
                      <span className="text-xs text-muted-foreground">
                        {fighter.weight_class} · {fighter.wins}-{fighter.losses}-{fighter.draws}
                      </span>
                    </div>
                  </CommandItem>
                ))}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>
    </div>
  );
}
