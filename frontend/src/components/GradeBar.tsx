interface GradeBarProps {
  probabilities: number[];
  predictedGrade: number;
}

const GRADE_LABELS = [
  "No DR",
  "Mild DR",
  "Moderate DR",
  "Severe DR",
  "Proliferative DR",
];

export default function GradeBar({ probabilities, predictedGrade }: GradeBarProps) {
  return (
    <div className="space-y-4 bg-gray-900/40 p-6 rounded-2xl border border-gray-800">
      <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
        Class Distribution Probabilities
      </h3>
      <div className="space-y-3">
        {GRADE_LABELS.map((label, idx) => {
          const prob = probabilities[idx] ?? 0.0;
          const percentage = (prob * 100).toFixed(1);
          const isPredicted = idx === predictedGrade;

          return (
            <div key={idx} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className={`font-medium ${isPredicted ? "text-teal-400" : "text-gray-300"}`}>
                  {label} {isPredicted && "• Predicted"}
                </span>
                <span className={`font-mono ${isPredicted ? "text-teal-400 font-bold" : "text-gray-400"}`}>
                  {percentage}%
                </span>
              </div>
              <div className="w-full bg-gray-800 h-2.5 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isPredicted ? "bg-teal-500 shadow-lg shadow-teal-500/30" : "bg-gray-600"
                  }`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
