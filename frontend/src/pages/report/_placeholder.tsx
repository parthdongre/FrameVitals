import type { ComponentType, ReactNode } from "react";
import { EmptyState } from "@/components/ui/EmptyState";
import type { TabComponentProps } from "./tabRegistry";

interface PlaceholderProps {
  title?: string;
  hint?: ReactNode;
}

/**
 * Tiny shared component used by the tab placeholders during phases 7-10.
 * Each phase replaces the relevant tab module with a real implementation;
 * having a single shared placeholder keeps the build green at every step.
 */
export function TabPlaceholder({ title, hint }: PlaceholderProps) {
  return (
    <EmptyState
      title={title ?? "Coming online in the next phase"}
      hint={
        hint ??
        "This tab is part of the upcoming phase. The route is wired and the registry knows about it; the panel body lands soon."
      }
    />
  );
}

/**
 * Wraps a placeholder into a default-exported tab module so the lazy loader
 * in `tabRegistry.ts` can pull it in without ceremony.
 */
export function makePlaceholderTab(
  title: string,
  hint?: ReactNode,
): ComponentType<TabComponentProps> {
  return function PlaceholderTab(_props: TabComponentProps) {
    return <TabPlaceholder title={title} hint={hint} />;
  };
}
