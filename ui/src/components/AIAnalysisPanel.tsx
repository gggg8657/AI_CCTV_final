import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { Brain, AlertTriangle, Activity, Shield, Eye, Camera, Loader2, RefreshCw } from "lucide-react";
import { Button } from "./ui/button";
import { cameras, events, stats, health, Event as ApiEvent } from "../lib/api";

export function AIAnalysisPanel() {
  const [totalCameras, setTotalCameras] = useState(0);
  const [activeCameras, setActiveCameras] = useState(0);
  const [totalEvents, setTotalEvents] = useState(0);
  const [unacked, setUnacked] = useState(0);
  const [avgVad, setAvgVad] = useState(0);
  const [recentEvents, setRecentEvents] = useState<ApiEvent[]>([]);
  const [systemHealthPct, setSystemHealthPct] = useState(0);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const [camList, evResult, summary, hp] = await Promise.all([
        cameras.list(),
        events.list({ limit: "8" }),
        stats.summary(1).catch(() => null),
        health().catch(() => null),
      ]);

      const active = camList.filter((c) => c.status === "active").length;
      setTotalCameras(camList.length);
      setActiveCameras(active);
      setRecentEvents(evResult.items);
      setTotalEvents(evResult.total);

      if (summary) {
        setUnacked(summary.unacknowledged);
        setAvgVad(summary.avg_vad_score);
      }

      if (hp?.status === "healthy") {
        const pct = camList.length > 0 ? Math.round((active / camList.length) * 100) : 0;
        setSystemHealthPct(pct);
      }
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const iv = setInterval(load, 10000);
    return () => clearInterval(iv);
  }, [load]);

  const fmtTime = (ts: string) => {
    try {
      return new Date(ts).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
    } catch {
      return ts;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              AI 분석 현황
            </CardTitle>
            <Button variant="ghost" size="sm" onClick={load}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Camera className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-semibold">{activeCameras}/{totalCameras}</p>
                <p className="text-sm text-muted-foreground">활성 카메라</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-orange-100 rounded-lg">
                <Activity className="h-5 w-5 text-orange-600" />
              </div>
              <div>
                <p className="text-2xl font-semibold">{totalEvents}</p>
                <p className="text-sm text-muted-foreground">탐지 이벤트</p>
              </div>
            </div>
          </div>

          {avgVad > 0 && (
            <div className="text-xs text-muted-foreground">
              평균 VAD 점수: <span className="font-mono">{avgVad.toFixed(4)}</span>
            </div>
          )}

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-blue-500" />
                <span className="text-sm">카메라 가동률</span>
              </div>
              <span className="text-sm text-muted-foreground">{systemHealthPct}%</span>
            </div>
            <Progress value={systemHealthPct} className="h-2" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            최근 이벤트
          </CardTitle>
        </CardHeader>
        <CardContent>
          {recentEvents.length === 0 ? (
            <p className="text-sm text-muted-foreground text-center py-4">탐지된 이벤트가 없습니다</p>
          ) : (
            <div className="space-y-3">
              {recentEvents.map((ev) => (
                <div key={ev.id} className="flex items-center gap-3 p-2 rounded-lg border">
                  <Activity className="h-4 w-4 text-orange-500" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm">
                      {ev.vlm_type || "이상 탐지"} — VAD {ev.vad_score.toFixed(2)}
                    </p>
                    <p className="text-xs text-muted-foreground">카메라 #{ev.camera_id}</p>
                  </div>
                  <Badge
                    variant={ev.acknowledged ? "secondary" : "destructive"}
                    className="text-xs"
                  >
                    {fmtTime(ev.timestamp)}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {unacked > 0 && (
        <Card className="border-red-200 bg-red-50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-700">
              <AlertTriangle className="h-5 w-5" />
              미확인 알림 ({unacked}건)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-red-600">확인되지 않은 이벤트가 {unacked}건 있습니다.</p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
