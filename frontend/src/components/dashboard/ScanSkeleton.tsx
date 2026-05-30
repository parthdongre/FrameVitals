import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export function ScanSkeleton() {
  return (
    <div className="min-h-screen bg-space-950 px-4 py-5 text-slate-100 sm:px-6 lg:px-8">
      <div className="mx-auto grid max-w-7xl gap-4 xl:grid-cols-12">
        <div className="hidden xl:col-span-3 xl:block">
          <Card className="h-[calc(100vh-2.5rem)] border-white/5 bg-white/[0.03] p-5 shadow-panel">
            <div className="space-y-4">
              <Skeleton className="h-6 w-32" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          </Card>
        </div>

        <div className="space-y-4 xl:col-span-9 xl:pl-0">
          <Skeleton className="h-24 w-full rounded-[1.6rem]" />
          <Skeleton className="h-24 w-full rounded-[1.6rem]" />

          <div className="grid gap-4 xl:grid-cols-12">
            <Card className="xl:col-span-5 border-white/5 bg-white/[0.03] shadow-panel">
              <CardHeader>
                <Skeleton className="h-6 w-44" />
                <Skeleton className="h-4 w-72" />
              </CardHeader>
              <CardContent className="space-y-4">
                <Skeleton className="h-48 w-full rounded-[1.5rem]" />
                <Skeleton className="h-11 w-full rounded-2xl" />
              </CardContent>
            </Card>

            <Card className="xl:col-span-7 border-white/5 bg-white/[0.03] shadow-panel">
              <CardHeader>
                <Skeleton className="h-6 w-52" />
                <Skeleton className="h-4 w-80" />
              </CardHeader>
              <CardContent className="space-y-4">
                <Skeleton className="h-[22rem] w-full rounded-[1.5rem]" />
                <div className="grid gap-3 sm:grid-cols-4">
                  <Skeleton className="h-20 rounded-2xl" />
                  <Skeleton className="h-20 rounded-2xl" />
                  <Skeleton className="h-20 rounded-2xl" />
                  <Skeleton className="h-20 rounded-2xl" />
                </div>
              </CardContent>
            </Card>

            <Skeleton className="h-28 rounded-[1.5rem] xl:col-span-4" />
            <Skeleton className="h-28 rounded-[1.5rem] xl:col-span-4" />
            <Skeleton className="h-28 rounded-[1.5rem] xl:col-span-4" />

            <Skeleton className="h-64 rounded-[1.6rem] xl:col-span-12" />
          </div>
        </div>
      </div>

      <div className="mx-auto mt-4 max-w-7xl">
        <div className="h-1 rounded-full bg-[linear-gradient(90deg,transparent,rgba(6,182,212,0.9),transparent)] bg-[length:220%_100%] animate-scan" />
      </div>
    </div>
  );
}