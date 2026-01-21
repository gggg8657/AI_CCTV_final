import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, TrendingDown, Users, Car, AlertTriangle, Camera } from "lucide-react";

const hourlyData = [
  { hour: "09:00", people: 12, vehicles: 8 },
  { hour: "10:00", people: 18, vehicles: 15 },
  { hour: "11:00", people: 25, vehicles: 22 },
  { hour: "12:00", people: 45, vehicles: 35 },
  { hour: "13:00", people: 38, vehicles: 28 },
  { hour: "14:00", people: 32, vehicles: 25 },
  { hour: "15:00", people: 28, vehicles: 20 }
];

const weeklyAlerts = [
  { day: "월", alerts: 3 },
  { day: "화", alerts: 1 },
  { day: "수", alerts: 4 },
  { day: "목", alerts: 2 },
  { day: "금", alerts: 5 },
  { day: "토", alerts: 1 },
  { day: "일", alerts: 2 }
];

const cameraStatus = [
  { name: "온라인", value: 15, color: "#22c55e" },
  { name: "오프라인", value: 1, color: "#ef4444" },
  { name: "점검중", value: 2, color: "#f59e0b" }
];

export function StatsDashboard() {
  return (
    <div className="space-y-6">
      {/* 주요 지표 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">일일 방문자</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">248</div>
            <p className="text-xs text-muted-foreground">
              <TrendingUp className="h-3 w-3 inline mr-1 text-green-500" />
              +12% 전일 대비
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">차량 출입</CardTitle>
            <Car className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">156</div>
            <p className="text-xs text-muted-foreground">
              <TrendingDown className="h-3 w-3 inline mr-1 text-red-500" />
              -3% 전일 대비
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">주간 알림</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">18</div>
            <p className="text-xs text-muted-foreground">
              <TrendingDown className="h-3 w-3 inline mr-1 text-green-500" />
              -25% 전주 대비
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">활성 카메라</CardTitle>
            <Camera className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">15/18</div>
            <p className="text-xs text-muted-foreground">
              정상 운영 중
            </p>
          </CardContent>
        </Card>
      </div>

      {/* 차트 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 시간별 활동 */}
        <Card>
          <CardHeader>
            <CardTitle>시간별 활동</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="hour" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="people" fill="#3b82f6" name="사람" />
                <Bar dataKey="vehicles" fill="#10b981" name="차량" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* 주간 알림 추이 */}
        <Card>
          <CardHeader>
            <CardTitle>주간 알림 추이</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={weeklyAlerts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="alerts" 
                  stroke="#ef4444" 
                  strokeWidth={2}
                  name="알림 수"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* 카메라 상태 */}
      <Card>
        <CardHeader>
          <CardTitle>카메라 상태 분포</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col lg:flex-row items-center gap-8">
            <div className="w-full lg:w-1/2">
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={cameraStatus}
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    dataKey="value"
                    label={({ name, value }) => `${name}: ${value}`}
                  >
                    {cameraStatus.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="w-full lg:w-1/2 space-y-4">
              {cameraStatus.map((status, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: status.color }}
                    />
                    <span className="text-sm">{status.name}</span>
                  </div>
                  <span className="text-sm font-medium">{status.value}대</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}