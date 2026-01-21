import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Progress } from "./ui/progress";
import { Brain, Users, Car, AlertTriangle, Activity, Shield, Eye } from "lucide-react";

interface AnalysisData {
  totalPeople: number;
  totalVehicles: number;
  activeAlerts: number;
  systemHealth: number;
  recentActivities: Array<{
    id: string;
    type: "person" | "vehicle" | "alert";
    message: string;
    time: string;
    camera: string;
  }>;
}

const mockAnalysisData: AnalysisData = {
  totalPeople: 6,
  totalVehicles: 8,
  activeAlerts: 1,
  systemHealth: 95,
  recentActivities: [
    {
      id: "1",
      type: "person",
      message: "3명의 사람이 감지됨",
      time: "13:45",
      camera: "정문 출입구"
    },
    {
      id: "2",
      type: "vehicle",
      message: "차량 진입 감지",
      time: "13:42",
      camera: "주차장 A구역"
    },
    {
      id: "3",
      type: "alert",
      message: "카메라 연결 끊어짐",
      time: "13:40",
      camera: "비상출구"
    },
    {
      id: "4",
      type: "person",
      message: "2명의 사람이 감지됨",
      time: "13:38",
      camera: "복도 2층"
    }
  ]
};

export function AIAnalysisPanel() {
  const getActivityIcon = (type: string) => {
    switch (type) {
      case "person":
        return <Users className="h-4 w-4 text-blue-500" />;
      case "vehicle":
        return <Car className="h-4 w-4 text-green-500" />;
      case "alert":
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const getActivityBadgeVariant = (type: string) => {
    switch (type) {
      case "person":
        return "default";
      case "vehicle":
        return "secondary";
      case "alert":
        return "destructive";
      default:
        return "outline";
    }
  };

  return (
    <div className="space-y-4">
      {/* AI 분석 요약 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            AI 분석 현황
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Users className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-semibold">{mockAnalysisData.totalPeople}</p>
                <p className="text-sm text-muted-foreground">감지된 사람</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <Car className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-semibold">{mockAnalysisData.totalVehicles}</p>
                <p className="text-sm text-muted-foreground">감지된 차량</p>
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-blue-500" />
                <span className="text-sm">시스템 상태</span>
              </div>
              <span className="text-sm text-muted-foreground">{mockAnalysisData.systemHealth}%</span>
            </div>
            <Progress value={mockAnalysisData.systemHealth} className="h-2" />
          </div>
        </CardContent>
      </Card>

      {/* 실시간 활동 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            실시간 활동
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {mockAnalysisData.recentActivities.map((activity) => (
              <div key={activity.id} className="flex items-center gap-3 p-2 rounded-lg border">
                {getActivityIcon(activity.type)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm">{activity.message}</p>
                  <p className="text-xs text-muted-foreground">{activity.camera}</p>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={getActivityBadgeVariant(activity.type)} className="text-xs">
                    {activity.time}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 활성 알림 */}
      {mockAnalysisData.activeAlerts > 0 && (
        <Card className="border-red-200 bg-red-50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-700">
              <AlertTriangle className="h-5 w-5" />
              활성 알림 ({mockAnalysisData.activeAlerts})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-red-500" />
                <span className="text-sm">카메라 연결 끊어짐 - 비상출구</span>
                <Badge variant="destructive" className="text-xs ml-auto">긴급</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}