import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";
import { Settings, Camera, Bell, Shield, Users, Palette } from "lucide-react";
import { useState } from "react";

export function SettingsPanel() {
  const [notifications, setNotifications] = useState(true);
  const [soundAlerts, setSoundAlerts] = useState(false);
  const [autoRecording, setAutoRecording] = useState(true);
  const [sensitivity, setSensitivity] = useState([75]);
  const [retentionDays, setRetentionDays] = useState([30]);

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Settings className="h-6 w-6" />
        <h2 className="text-xl font-semibold">시스템 설정</h2>
      </div>

      <Tabs defaultValue="cameras" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="cameras" className="flex items-center gap-2">
            <Camera className="h-4 w-4" />
            카메라
          </TabsTrigger>
          <TabsTrigger value="notifications" className="flex items-center gap-2">
            <Bell className="h-4 w-4" />
            알림
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center gap-2">
            <Shield className="h-4 w-4" />
            보안
          </TabsTrigger>
          <TabsTrigger value="appearance" className="flex items-center gap-2">
            <Palette className="h-4 w-4" />
            화면
          </TabsTrigger>
        </TabsList>

        <TabsContent value="cameras" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>카메라 관리</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* AI 감지 설정 */}
              <div className="space-y-4">
                <h3 className="font-medium">AI 감지 설정</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>사람 감지 민감도</Label>
                    <Slider
                      value={sensitivity}
                      onValueChange={setSensitivity}
                      max={100}
                      step={5}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">현재: {sensitivity[0]}%</p>
                  </div>
                  <div className="space-y-2">
                    <Label>녹화 보관 기간</Label>
                    <Slider
                      value={retentionDays}
                      onValueChange={setRetentionDays}
                      max={90}
                      min={7}
                      step={1}
                      className="w-full"
                    />
                    <p className="text-xs text-muted-foreground">현재: {retentionDays[0]}일</p>
                  </div>
                </div>
              </div>

              {/* 자동 녹화 */}
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>자동 녹화</Label>
                  <p className="text-sm text-muted-foreground">움직임 감지 시 자동으로 녹화를 시작합니다</p>
                </div>
                <Switch
                  checked={autoRecording}
                  onCheckedChange={setAutoRecording}
                />
              </div>

              {/* 카메라 목록 */}
              <div className="space-y-3">
                <h3 className="font-medium">연결된 카메라</h3>
                <div className="space-y-2">
                  {[
                    { name: "정문 출입구", status: "online", ip: "192.168.1.101" },
                    { name: "주차장 A구역", status: "online", ip: "192.168.1.102" },
                    { name: "복도 2층", status: "online", ip: "192.168.1.103" },
                    { name: "비상출구", status: "offline", ip: "192.168.1.104" }
                  ].map((camera, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <Camera className="h-4 w-4" />
                        <div>
                          <p className="text-sm font-medium">{camera.name}</p>
                          <p className="text-xs text-muted-foreground">{camera.ip}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={camera.status === "online" ? "default" : "destructive"}>
                          {camera.status === "online" ? "온라인" : "오프라인"}
                        </Badge>
                        <Button variant="outline" size="sm">설정</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>알림 설정</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>푸시 알림</Label>
                  <p className="text-sm text-muted-foreground">중요한 이벤트 발생 시 알림을 받습니다</p>
                </div>
                <Switch
                  checked={notifications}
                  onCheckedChange={setNotifications}
                />
              </div>

              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>음성 알림</Label>
                  <p className="text-sm text-muted-foreground">알림 발생 시 소리로 알려줍니다</p>
                </div>
                <Switch
                  checked={soundAlerts}
                  onCheckedChange={setSoundAlerts}
                />
              </div>

              <div className="space-y-2">
                <Label>알림 유형</Label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {[
                    { name: "사람 감지", enabled: true },
                    { name: "차량 감지", enabled: true },
                    { name: "침입 감지", enabled: true },
                    { name: "카메라 오류", enabled: true },
                    { name: "시스템 점검", enabled: false },
                    { name: "저장공간 부족", enabled: true }
                  ].map((notification, index) => (
                    <div key={index} className="flex items-center justify-between p-2 border rounded">
                      <span className="text-sm">{notification.name}</span>
                      <Switch defaultChecked={notification.enabled} />
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>보안 설정</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>관리자 패스워드 변경</Label>
                  <Input type="password" placeholder="현재 패스워드" />
                  <Input type="password" placeholder="새 패스워드" />
                  <Input type="password" placeholder="새 패스워드 확인" />
                  <Button className="w-full">패스워드 변경</Button>
                </div>

                <div className="space-y-2">
                  <Label>접근 권한 관리</Label>
                  <div className="space-y-2">
                    {[
                      { name: "관리자", role: "admin", access: "전체" },
                      { name: "보안팀", role: "security", access: "모니터링" },
                      { name: "운영팀", role: "operator", access: "제한적" }
                    ].map((user, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex items-center gap-3">
                          <Users className="h-4 w-4" />
                          <div>
                            <p className="text-sm font-medium">{user.name}</p>
                            <p className="text-xs text-muted-foreground">{user.role}</p>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{user.access}</Badge>
                          <Button variant="outline" size="sm">편집</Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="appearance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>화면 설정</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>테마</Label>
                <Select defaultValue="system">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="light">라이트 모드</SelectItem>
                    <SelectItem value="dark">다크 모드</SelectItem>
                    <SelectItem value="system">시스템 설정 따름</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>카메라 그리드 레이아웃</Label>
                <Select defaultValue="2x2">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1x1">1x1 (단일 화면)</SelectItem>
                    <SelectItem value="2x2">2x2 (4개 화면)</SelectItem>
                    <SelectItem value="3x3">3x3 (9개 화면)</SelectItem>
                    <SelectItem value="4x4">4x4 (16개 화면)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>언어</Label>
                <Select defaultValue="ko">
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="ko">한국어</SelectItem>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="ja">日本語</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}