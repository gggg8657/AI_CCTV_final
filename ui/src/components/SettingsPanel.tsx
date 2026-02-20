import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Switch } from "./ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Slider } from "./ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";
import { Settings, Camera as CameraIcon, Bell, Shield, Palette, Plus, Trash2, Loader2, RefreshCw } from "lucide-react";
import { useState, useEffect, useCallback } from "react";
import { cameras, Camera } from "../lib/api";

function AddCameraForm({ onAdded }: { onAdded: () => void }) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    name: "",
    source_type: "dummy",
    source_path: "synthetic",
    location: "",
    vad_model: "mnad",
    vad_threshold: 0.5,
    enable_vlm: true,
    enable_agent: true,
  });

  const set = (key: string, value: string | number | boolean) => {
    setForm((p) => {
      const next = { ...p, [key]: value };
      if (key === "source_type") {
        const defaults: Record<string, string> = {
          dummy: "synthetic",
          webcam: "0",
          rtsp: "rtsp://",
          file: "",
        };
        next.source_path = defaults[value as string] ?? "";
      }
      return next;
    });
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) { setError("카메라 이름을 입력하세요"); return; }
    setLoading(true);
    setError("");
    try {
      await cameras.create(form);
      setForm({ name: "", source_type: "dummy", source_path: "synthetic", location: "", vad_model: "mnad", vad_threshold: 0.5, enable_vlm: true, enable_agent: true });
      setOpen(false);
      onAdded();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!open) {
    return (
      <Button variant="outline" className="w-full" onClick={() => setOpen(true)}>
        <Plus className="h-4 w-4 mr-2" />
        카메라 추가
      </Button>
    );
  }

  return (
    <Card>
      <CardContent className="pt-4">
        <form onSubmit={submit} className="space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="space-y-1">
              <Label className="text-xs">카메라 이름 *</Label>
              <Input placeholder="예: 정문 카메라" value={form.name} onChange={(e) => set("name", e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">위치</Label>
              <Input placeholder="예: Main Entrance" value={form.location} onChange={(e) => set("location", e.target.value)} />
            </div>
            <div className="space-y-1">
              <Label className="text-xs">소스 타입</Label>
              <Select value={form.source_type} onValueChange={(v: string) => set("source_type", v)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="dummy">Dummy (테스트)</SelectItem>
                  <SelectItem value="rtsp">RTSP 스트림</SelectItem>
                  <SelectItem value="file">비디오 파일</SelectItem>
                  <SelectItem value="webcam">웹캠</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">소스 경로</Label>
              <Input
                placeholder={
                  form.source_type === "webcam" ? "장치 번호 (0, 1, ...)" :
                  form.source_type === "rtsp" ? "rtsp://192.168.1.100:554/stream" :
                  form.source_type === "file" ? "/path/to/video.mp4" :
                  "synthetic"
                }
                value={form.source_path}
                onChange={(e) => set("source_path", e.target.value)}
              />
              {form.source_type === "webcam" && (
                <p className="text-xs text-muted-foreground">기본 웹캠은 0번입니다</p>
              )}
            </div>
            <div className="space-y-1">
              <Label className="text-xs">VAD 모델</Label>
              <Select value={form.vad_model} onValueChange={(v: string) => set("vad_model", v)}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="mnad">MNAD</SelectItem>
                  <SelectItem value="stae">STAE</SelectItem>
                  <SelectItem value="stead">STEAD</SelectItem>
                  <SelectItem value="memae">MemAE</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="text-xs">VAD 임계값: {form.vad_threshold}</Label>
              <Slider value={[form.vad_threshold]} onValueChange={([v]: number[]) => set("vad_threshold", v)} min={0.1} max={1.0} step={0.05} />
            </div>
          </div>
          <div className="flex items-center gap-6">
            <label className="flex items-center gap-2 text-sm">
              <Switch checked={form.enable_vlm} onCheckedChange={(v: boolean) => set("enable_vlm", v)} />
              VLM 분석
            </label>
            <label className="flex items-center gap-2 text-sm">
              <Switch checked={form.enable_agent} onCheckedChange={(v: boolean) => set("enable_agent", v)} />
              Agent 대응
            </label>
          </div>
          {error && <p className="text-sm text-red-500">{error}</p>}
          <div className="flex gap-2">
            <Button type="submit" disabled={loading} className="flex-1">
              {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Plus className="h-4 w-4 mr-2" />}
              등록
            </Button>
            <Button type="button" variant="outline" onClick={() => setOpen(false)}>취소</Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

export function SettingsPanel() {
  const [notifications, setNotifications] = useState(true);
  const [soundAlerts, setSoundAlerts] = useState(false);
  const [cams, setCams] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<Record<number, boolean>>({});

  const load = useCallback(async () => {
    try {
      const list = await cameras.list();
      setCams(list);
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (id: number) => {
    if (!confirm("이 카메라를 삭제하시겠습니까?")) return;
    setDeleting((p) => ({ ...p, [id]: true }));
    try {
      await cameras.delete(id);
      await load();
    } catch {
      // ignore
    } finally {
      setDeleting((p) => ({ ...p, [id]: false }));
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <Settings className="h-6 w-6" />
        <h2 className="text-xl font-semibold">시스템 설정</h2>
      </div>

      <Tabs defaultValue="cameras" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="cameras" className="flex items-center gap-2">
            <CameraIcon className="h-4 w-4" />
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
              <div className="flex items-center justify-between">
                <CardTitle>카메라 관리</CardTitle>
                <Button variant="ghost" size="sm" onClick={load}>
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <AddCameraForm onAdded={load} />

              {loading ? (
                <div className="flex justify-center py-8">
                  <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                </div>
              ) : cams.length === 0 ? (
                <p className="text-center text-muted-foreground py-6">등록된 카메라가 없습니다</p>
              ) : (
                <div className="space-y-2">
                  {cams.map((cam) => (
                    <div key={cam.id} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <CameraIcon className="h-4 w-4" />
                        <div>
                          <p className="text-sm font-medium">{cam.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {cam.location || cam.source_path} &middot; {cam.vad_model} &middot; threshold={cam.vad_threshold}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {cam.enable_vlm && <Badge variant="outline" className="text-xs">VLM</Badge>}
                        {cam.enable_agent && <Badge variant="outline" className="text-xs">Agent</Badge>}
                        <Badge variant={cam.status === "active" ? "default" : "secondary"}>
                          {cam.status === "active" ? "활성" : "비활성"}
                        </Badge>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0 text-red-500 hover:text-red-700"
                          onClick={() => handleDelete(cam.id)}
                          disabled={deleting[cam.id]}
                        >
                          {deleting[cam.id] ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
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
                <Switch checked={notifications} onCheckedChange={setNotifications} />
              </div>
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>음성 알림</Label>
                  <p className="text-sm text-muted-foreground">알림 발생 시 소리로 알려줍니다</p>
                </div>
                <Switch checked={soundAlerts} onCheckedChange={setSoundAlerts} />
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
                    { name: "저장공간 부족", enabled: true },
                  ].map((n, i) => (
                    <div key={i} className="flex items-center justify-between p-2 border rounded">
                      <span className="text-sm">{n.name}</span>
                      <Switch defaultChecked={n.enabled} />
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
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">
                  패스워드 변경 기능은 현재 준비 중입니다. 향후 업데이트에서 지원될 예정입니다.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="appearance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>화면 설정</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>테마</Label>
                <Select defaultValue="system">
                  <SelectTrigger><SelectValue /></SelectTrigger>
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
                  <SelectTrigger><SelectValue /></SelectTrigger>
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
                  <SelectTrigger><SelectValue /></SelectTrigger>
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
