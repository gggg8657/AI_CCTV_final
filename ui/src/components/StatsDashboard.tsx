import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import { AlertTriangle, Camera, Activity, Shield, Loader2 } from "lucide-react";
import { useState, useEffect } from "react";
import { cameras, events, health as fetchHealth } from "../lib/api";
import type { Camera as CameraType, Event } from "../lib/api";

export function StatsDashboard() {
  const [cams, setCams] = useState<CameraType[]>([]);
  const [evts, setEvts] = useState<{ items: Event[]; total: number }>({ items: [], total: 0 });
  const [healthData, setHealthData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      cameras.list().catch(() => []),
      events.list({ limit: "50" }).catch(() => ({ items: [], total: 0 })),
      fetchHealth().catch(() => null),
    ]).then(([c, e, h]) => {
      setCams(c);
      setEvts(e);
      setHealthData(h);
      setLoading(false);
    });
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const activeCams = cams.filter((c) => c.status === "active").length;
  const inactiveCams = cams.length - activeCams;
  const unacked = evts.items.filter((e) => !e.acknowledged).length;

  const cameraStatus = [
    { name: "활성", value: activeCams, color: "#22c55e" },
    { name: "비활성", value: inactiveCams, color: "#ef4444" },
  ].filter((d) => d.value > 0);

  const typeCounts: Record<string, number> = {};
  evts.items.forEach((e) => {
    const t = e.vlm_type || "unknown";
    typeCounts[t] = (typeCounts[t] || 0) + 1;
  });
  const typeData = Object.entries(typeCounts).map(([name, value]) => ({ name, value }));
  const TYPE_COLORS = ["#3b82f6", "#ef4444", "#f59e0b", "#8b5cf6", "#10b981", "#ec4899"];

  const dummyFlags = healthData?.pipeline?.dummy_flags;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">등록 카메라</CardTitle>
            <Camera className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{activeCams}/{cams.length}</div>
            <p className="text-xs text-muted-foreground">활성 / 전체</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">총 이벤트</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{evts.total}</div>
            <p className="text-xs text-muted-foreground">미확인 {unacked}건</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">시스템</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{healthData?.status === "healthy" ? "정상" : "점검"}</div>
            <p className="text-xs text-muted-foreground">{healthData?.service}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">파이프라인</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dummyFlags ? "Dummy" : "Real"}
            </div>
            <p className="text-xs text-muted-foreground">
              {dummyFlags && `VAD:${dummyFlags.vad ? "D" : "R"} VLM:${dummyFlags.vlm ? "D" : "R"} Agent:${dummyFlags.agent ? "D" : "R"}`}
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>카메라 상태</CardTitle>
          </CardHeader>
          <CardContent>
            {cameraStatus.length > 0 ? (
              <div className="flex flex-col lg:flex-row items-center gap-8">
                <div className="w-full lg:w-1/2">
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie data={cameraStatus} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                        {cameraStatus.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="w-full lg:w-1/2 space-y-4">
                  {cameraStatus.map((s, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: s.color }} />
                        <span className="text-sm">{s.name}</span>
                      </div>
                      <span className="text-sm font-medium">{s.value}대</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="text-center text-muted-foreground py-8">카메라 데이터 없음</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>이벤트 유형 분포</CardTitle>
          </CardHeader>
          <CardContent>
            {typeData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie data={typeData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, value }) => `${name}: ${value}`}>
                    {typeData.map((_, i) => (
                      <Cell key={i} fill={TYPE_COLORS[i % TYPE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-center text-muted-foreground py-8">이벤트 데이터 없음</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
