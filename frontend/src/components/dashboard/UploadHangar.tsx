import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { CloudUpload, FileUp, Orbit, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

interface UploadHangarProps {
  onFileSelected?: (file: File) => void;
}

export function UploadHangar({ onFileSelected }: UploadHangarProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileLabel, setFileLabel] = useState<string | null>(null);

  const handleFile = (file: File | null) => {
    if (!file) {
      return;
    }

    setFileLabel(file.name);
    onFileSelected?.(file);
  };

  return (
    <Card className="relative overflow-hidden border-dashed border-cyan-400/20 bg-white/[0.03] shadow-panel">
      <motion.div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(6,182,212,0.16),transparent_60%)]"
        animate={{ opacity: isDragging ? 1 : 0.25, scale: isDragging ? 1.08 : 1 }}
        transition={{ duration: 0.35 }}
      />
      {isDragging ? (
        <motion.div
          aria-hidden="true"
          className="pointer-events-none absolute inset-y-0 left-0 w-1/3 bg-[linear-gradient(90deg,transparent,rgba(6,182,212,0.8),transparent)] opacity-60 blur-2xl"
          animate={{ x: ["-120%", "120%"] }}
          transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
        />
      ) : null}

      <CardHeader className="relative">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2 text-slate-50">
              <CloudUpload className="h-4 w-4 text-cyan-300" />
              Dataset Upload
            </CardTitle>
            <CardDescription className="mt-1 text-slate-400">
              Upload a dataset and the analysis pipeline will generate the report.
            </CardDescription>
          </div>
          <div className="rounded-2xl border border-cyan-400/15 bg-cyan-500/10 p-3 text-cyan-200 shadow-halo">
            <Orbit className="h-5 w-5 animate-orbit" />
          </div>
        </div>
      </CardHeader>

      <CardContent className="relative space-y-4">
        <label
          onDragEnter={() => setIsDragging(true)}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            handleFile(event.dataTransfer.files?.[0] ?? null);
          }}
          className="group flex cursor-pointer flex-col items-center justify-center gap-4 rounded-[1.5rem] border border-dashed border-cyan-400/30 bg-space-900/55 px-5 py-8 text-center transition-all duration-300 hover:border-cyan-300/50 hover:shadow-warp"
          style={{
            filter: isDragging ? "drop-shadow(0 0 10px #06b6d4)" : "none",
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".csv,.tsv,.json,.xlsx,.xls"
            className="hidden"
            onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
          />

          <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-cyan-400/20 bg-cyan-500/10 text-cyan-200 shadow-halo">
            <Sparkles className="h-6 w-6" />
          </div>

          <div>
            <p className="text-sm font-semibold text-slate-100">Drop a file or browse for a dataset</p>
            <p className="mt-1 text-xs text-slate-500">
              CSV, TSV, JSON, XLSX, XLS. Supported formats are shown below.
            </p>
          </div>

          <Button
            type="button"
            variant="trace"
            onClick={() => inputRef.current?.click()}
            className="min-w-44"
          >
            <FileUp className="h-4 w-4" />
            Select File
          </Button>
        </label>

        <div className="flex items-center justify-between rounded-2xl border border-white/5 bg-white/[0.02] px-4 py-3 text-sm">
          <div>
            <p className="text-slate-500">Selected file</p>
            <p className="font-mono text-slate-100">{fileLabel ?? "No file uploaded"}</p>
          </div>
          <div className="text-right">
            <p className="text-slate-500">Upload state</p>
            <p className="font-mono text-cyan-200">{isDragging ? "Dragging" : "Ready"}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}