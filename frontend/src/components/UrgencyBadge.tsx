// UrgencyBadge.tsx
// ----------------
// Why it exists:
//   Displays the clinical urgency level with appropriate color coding.
//   Color-coded urgency is critical for at-a-glance clinical triage.
//
// What it does:
//   - Maps urgency string → Tailwind CSS color class (via index.css).
//   - Renders a pill-shaped badge.
//
// Imported by:
//   - src/pages/ResultPage.tsx
//   - src/pages/HistoryPage.tsx

interface UrgencyBadgeProps {
  urgency: "routine" | "soon" | "urgent" | "emergency";
}

const URGENCY_LABELS: Record<string, string> = {
  routine: "Routine",
  soon: "See Soon (3 months)",
  urgent: "Urgent (1 month)",
  emergency: "Emergency — Immediate",
};

export default function UrgencyBadge({ urgency }: UrgencyBadgeProps) {
  // TODO: render <span className={`badge-${urgency} px-3 py-1 rounded-full text-sm font-medium`}>
  // {URGENCY_LABELS[urgency]}
  return <span className={`badge-${urgency} px-3 py-1 rounded-full text-sm font-medium`}>{URGENCY_LABELS[urgency]}</span>;
}
