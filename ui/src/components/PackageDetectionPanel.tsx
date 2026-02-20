import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "./ui/table";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import {
  Package,
  AlertTriangle,
  CheckCircle2,
  Clock,
  MapPin,
  RefreshCw,
  Eye,
  FileVideo,
  Info,
} from "lucide-react";

const formatDate = (dateString: string, format: string = "MM/dd HH:mm:ss") => {
  const date = new Date(dateString);
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  if (format === "yyyy-MM-dd HH:mm:ss") return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  return `${month}/${day} ${hours}:${minutes}:${seconds}`;
};

interface PackageInfo {
  package_id: string;
  status: "present" | "missing" | "stolen";
  first_seen: string;
  last_seen: string;
  current_position: [number, number, number, number];
  detection_count: number;
  camera_id: number;
}

interface PackageCount {
  total: number;
  present: number;
  missing: number;
  stolen: number;
}

interface ActivityEvent {
  event_type: string;
  timestamp: string;
  package_id: string;
  camera_id: number;
  [key: string]: any;
}

const API_BASE = import.meta.env.VITE_API_BASE || "";

export function PackageDetectionPanel() {
  const [packageCount, setPackageCount] = useState<PackageCount>({ total: 0, present: 0, missing: 0, stolen: 0 });
  const [packages, setPackages] = useState<PackageInfo[]>([]);
  const [activities, setActivities] = useState<ActivityEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiAvailable, setApiAvailable] = useState<boolean | null>(null);

  const fetchPackageCount = async () => {
    const res = await fetch(`${API_BASE}/api/v1/agent/function/get_package_count`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setPackageCount(data);
  };

  const fetchPackages = async () => {
    const res = await fetch(`${API_BASE}/api/v1/agent/function/get_packages`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setPackages(data);
  };

  const fetchActivities = async () => {
    const res = await fetch(`${API_BASE}/api/v1/agent/function/get_activities`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setActivities(data);
  };

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await Promise.all([fetchPackageCount(), fetchPackages(), fetchActivities()]);
      setApiAvailable(true);
    } catch {
      setApiAvailable(false);
      setPackageCount({ total: 0, present: 0, missing: 0, stolen: 0 });
      setPackages([]);
      setActivities([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "present":
        return <Badge variant="default" className="bg-green-500"><CheckCircle2 className="h-3 w-3 mr-1" />감지됨</Badge>;
      case "missing":
        return <Badge variant="default" className="bg-yellow-500"><Clock className="h-3 w-3 mr-1" />사라짐</Badge>;
      case "stolen":
        return <Badge variant="destructive"><AlertTriangle className="h-3 w-3 mr-1" />도난</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case "PackageDetectedEvent": return <Package className="h-4 w-4 text-green-500" />;
      case "PackageDisappearedEvent": return <Clock className="h-4 w-4 text-yellow-500" />;
      case "TheftDetectedEvent": return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default: return <Package className="h-4 w-4" />;
    }
  };

  const getEventLabel = (eventType: string) => {
    switch (eventType) {
      case "PackageDetectedEvent": return "패키지 감지";
      case "PackageDisappearedEvent": return "패키지 사라짐";
      case "TheftDetectedEvent": return "도난 감지";
      default: return eventType;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">패키지 감지 및 도난 방지</h2>
          <p className="text-muted-foreground">실시간 패키지 감지 상태와 도난 이벤트를 모니터링합니다</p>
        </div>
        <Button onClick={refreshAll} disabled={loading} variant="outline">
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          새로고침
        </Button>
      </div>

      {apiAvailable === false && (
        <Alert>
          <Info className="h-4 w-4" />
          <AlertTitle>API 미연결</AlertTitle>
          <AlertDescription>
            패키지 감지 API 엔드포인트가 아직 구현되지 않았거나 서버가 실행 중이지 않습니다.
            Agent 모듈이 활성화되면 자동으로 데이터가 표시됩니다.
          </AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>오류</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">전체 패키지</CardTitle>
            <Package className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{packageCount.total}</div>
            <p className="text-xs text-muted-foreground">현재 추적 중</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">감지됨</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">{packageCount.present}</div>
            <p className="text-xs text-muted-foreground">정상 상태</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">사라짐</CardTitle>
            <Clock className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-500">{packageCount.missing}</div>
            <p className="text-xs text-muted-foreground">추적 중</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">도난</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">{packageCount.stolen}</div>
            <p className="text-xs text-muted-foreground">확인됨</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="packages" className="space-y-4">
        <TabsList>
          <TabsTrigger value="packages">패키지 목록</TabsTrigger>
          <TabsTrigger value="activities">활동 로그</TabsTrigger>
        </TabsList>

        <TabsContent value="packages" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>추적 중인 패키지</CardTitle>
              <CardDescription>현재 감지되고 있는 모든 패키지의 상태</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>패키지 ID</TableHead>
                    <TableHead>상태</TableHead>
                    <TableHead>위치</TableHead>
                    <TableHead>감지 횟수</TableHead>
                    <TableHead>최초 감지</TableHead>
                    <TableHead>최종 감지</TableHead>
                    <TableHead>액션</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {packages.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={7} className="text-center text-muted-foreground">
                        {apiAvailable === false ? "API 미연결 — 데이터 없음" : "감지된 패키지가 없습니다"}
                      </TableCell>
                    </TableRow>
                  ) : (
                    packages.map((pkg) => (
                      <TableRow key={pkg.package_id}>
                        <TableCell className="font-mono">{pkg.package_id}</TableCell>
                        <TableCell>{getStatusBadge(pkg.status)}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1 text-xs">
                            <MapPin className="h-3 w-3" />
                            ({pkg.current_position[0]}, {pkg.current_position[1]})
                          </div>
                        </TableCell>
                        <TableCell>{pkg.detection_count}</TableCell>
                        <TableCell className="text-xs">{formatDate(pkg.first_seen)}</TableCell>
                        <TableCell className="text-xs">{formatDate(pkg.last_seen)}</TableCell>
                        <TableCell>
                          <Button variant="ghost" size="sm"><Eye className="h-4 w-4" /></Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="activities" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>최근 활동 로그</CardTitle>
              <CardDescription>패키지 감지, 사라짐, 도난 이벤트 기록</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {activities.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    {apiAvailable === false ? "API 미연결 — 데이터 없음" : "활동 로그가 없습니다"}
                  </div>
                ) : (
                  activities.map((activity, idx) => (
                    <div key={idx} className="flex items-start gap-4 p-4 border rounded-lg hover:bg-accent/50 transition-colors">
                      <div className="mt-1">{getEventIcon(activity.event_type)}</div>
                      <div className="flex-1 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold">{getEventLabel(activity.event_type)}</span>
                          <Badge variant="outline" className="text-xs">{activity.package_id}</Badge>
                          {activity.event_type === "TheftDetectedEvent" && (
                            <Badge variant="destructive" className="text-xs">긴급</Badge>
                          )}
                        </div>
                        <div className="text-sm text-muted-foreground">
                          {formatDate(activity.timestamp, "yyyy-MM-dd HH:mm:ss")}
                        </div>
                        {activity.event_type === "TheftDetectedEvent" && activity.evidence_frame_paths && (
                          <div className="flex items-center gap-2 mt-2">
                            <FileVideo className="h-4 w-4 text-muted-foreground" />
                            <span className="text-xs text-muted-foreground">
                              증거 영상: {activity.evidence_frame_paths.length}개
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
