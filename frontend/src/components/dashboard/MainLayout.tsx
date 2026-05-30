import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface MainLayoutProps {
  sidebar: React.ReactNode;
  children: React.ReactNode;
}

export function MainLayout({ sidebar, children }: MainLayoutProps) {
  return (
    <div className="relative min-h-screen overflow-hidden bg-space-950 text-slate-100">
      <div className="fixed inset-0 -z-20 bg-radial-cockpit" />
      <div className="fixed inset-0 -z-10 bg-void-grid bg-[size:72px_72px] opacity-40 [mask-image:linear-gradient(to_bottom,transparent,black_10%,black_90%,transparent)]" />
      <div className="pointer-events-none fixed inset-x-0 top-0 -z-10 h-40 bg-[linear-gradient(180deg,rgba(6,182,212,0.07),transparent)]" />

      <div className="relative mx-auto flex min-h-screen w-full max-w-[1920px]">
        <aside className="fixed inset-y-0 left-0 z-20 hidden w-[20.5rem] border-r border-white/5 bg-space-950/80 backdrop-blur-3xl xl:flex">
          <motion.div
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
            className={cn("flex h-full w-full flex-col overflow-y-auto p-5")}
          >
            {sidebar}
          </motion.div>
        </aside>

        <main className="relative flex min-h-screen w-full flex-1 flex-col xl:pl-[20.5rem]">
          {children}
        </main>
      </div>
    </div>
  );
}