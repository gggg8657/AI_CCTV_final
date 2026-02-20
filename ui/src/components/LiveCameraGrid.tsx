import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Camera as CameraIcon, AlertTriangle, Play, Pause, Maximize, RefreshCw, Loader2 } from "lucide-react";
import { useState, useEffect, useCallback } from "react";
import { cameras, Camera } from "../lib/api";

export function LiveCameraGrid() {
  const [cams, setCams] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [actionLoading, setActionLoading] = useState<Record<number, boolean>>({});

  const load = useCallback(async () => {
    try {
      const list = await cameras.list();
      setCams(list);
      setError("");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, [load]);

  const toggleCamera = async (cam: Camera) => {
    setActionLoading((p) => ({ ...p, [cam.id]: true }));
    try {
      if (cam.status === "active") {
        await cameras.stop(cam.id);
      } else {
        await cameras.start(cam.id);
      }
      await load();
    } catch {
      // ignore
    } finally {
      setActionLoading((p) => ({ ...p, [cam.id]: false }));
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center space-y-4 py-8">
        <p className="text-red-500">{error}</p>
        <Button variant="outline" onClick={load}>
          <RefreshCw className="h-4 w-4 mr-2" />
          재시도
        </Button>
      </div>
    );
  }

  if (cams.length === 0) {
    return (
      <Card>
        <CardContent className="flex flex-col items-center justify-center py-12 text-muted-foreground">
          <CameraIcon className="h-12 w-12 mb-4" />
          <p className="text-lg font-medium">등록된 카메라 없음</p>
          <p className="text-sm">설정에서 카메라를 추가하세요</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-4">
      {cams.map((cam) => {
        const isActive = cam.status === "active";
        return (
          <Card key={cam.id} className="overflow-hidden">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CameraIcon className="h-4 w-4" />
                  <CardTitle className="text-sm">{cam.name}</CardTitle>
                </div>
                <Badge variant={isActive ? "default" : "destructive"} className="text-xs">
                  {isActive ? "활성" : "비활성"}
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground">{cam.location || cam.source_path}</p>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="relative aspect-video bg-muted rounded-lg overflow-hidden flex items-center justify-center">
                {isActive ? (
                  <>
                    <div className="absolute inset-0 bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center">
                      <CameraIcon className="h-8 w-8 text-muted-foreground" />
                    </div>
                    <div className="absolute top-2 left-2">
                      <Badge variant="destructive" className="text-xs">
                        <div className="w-2 h-2 bg-white rounded-full mr-1 animate-pulse" />
                        LIVE
                      </Badge>
                    </div>
                  </>
                ) : (
                  <CameraIcon className="h-8 w-8 text-muted-foreground" />
                )}
                <div className="absolute bottom-2 right-2 flex gap-1">
                  <Button
                    size="sm"
                    variant="secondary"
                    className="h-6 w-6 p-0"
                    onClick={() => toggleCamera(cam)}
                    disabled={actionLoading[cam.id]}
                  >
                    {actionLoading[cam.id] ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : isActive ? (
                      <Pause className="h-3 w-3" />
                    ) : (
                      <Play className="h-3 w-3" />
                    )}
                  </Button>
                </div>
              </div>
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-3">
                  <span className="text-muted-foreground">VAD: {cam.vad_model}</span>
                  {cam.enable_vlm && <Badge variant="outline" className="text-xs">VLM</Badge>}
                  {cam.enable_agent && <Badge variant="outline" className="text-xs">Agent</Badge>}
                </div>
                <span className="text-muted-foreground">#{cam.id}</span>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
