import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import { Camera, Users, Car, AlertTriangle, Play, Pause, Maximize } from "lucide-react";
import { useState } from "react";

interface CameraData {
  id: string;
  name: string;
  location: string;
  status: "online" | "offline";
  imageUrl: string;
  detections: {
    people: number;
    vehicles: number;
    alerts: number;
  };
}

const mockCameras: CameraData[] = [
  {
    id: "cam-01",
    name: "정문 출입구",
    location: "1층 메인 로비",
    status: "online",
    imageUrl: "https://images.unsplash.com/photo-1673551799281-3a08f4dba4f5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBvZmZpY2UlMjBidWlsZGluZyUyMGVudHJhbmNlfGVufDF8fHx8MTc1ODY4OTQ5OHww&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    detections: { people: 3, vehicles: 0, alerts: 0 }
  },
  {
    id: "cam-02", 
    name: "주차장 A구역",
    location: "지하 1층",
    status: "online",
    imageUrl: "https://images.unsplash.com/photo-1643369654929-ae1c477beb9a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwYXJraW5nJTIwZ2FyYWdlJTIwc2VjdXJpdHl8ZW58MXx8fHwxNzU4NjM1OTE3fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    detections: { people: 1, vehicles: 8, alerts: 0 }
  },
  {
    id: "cam-03",
    name: "복도 2층",
    location: "2층 동쪽 복도", 
    status: "online",
    imageUrl: "https://images.unsplash.com/photo-1589305799001-57124aa2fe3a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxoYWxsd2F5JTIwY29ycmlkb3IlMjBzZWN1cml0eXxlbnwxfHx8fDE3NTg2ODk1MDJ8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    detections: { people: 2, vehicles: 0, alerts: 0 }
  },
  {
    id: "cam-04",
    name: "비상출구",
    location: "3층 서쪽",
    status: "offline",
    imageUrl: "https://images.unsplash.com/photo-1665848383782-1ea74efde68f?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzZWN1cml0eSUyMGNhbWVyYSUyMHN1cnZlaWxsYW5jZXxlbnwxfHx8fDE3NTg2NTM2NzV8MA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral",
    detections: { people: 0, vehicles: 0, alerts: 1 }
  }
];

export function LiveCameraGrid() {
  const [playingCameras, setPlayingCameras] = useState<Set<string>>(new Set(["cam-01", "cam-02", "cam-03"]));

  const toggleCamera = (cameraId: string) => {
    setPlayingCameras(prev => {
      const newSet = new Set(prev);
      if (newSet.has(cameraId)) {
        newSet.delete(cameraId);
      } else {
        newSet.add(cameraId);
      }
      return newSet;
    });
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-2 gap-4">
      {mockCameras.map((camera) => (
        <Card key={camera.id} className="overflow-hidden">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Camera className="h-4 w-4" />
                <CardTitle className="text-sm">{camera.name}</CardTitle>
              </div>
              <Badge 
                variant={camera.status === "online" ? "default" : "destructive"}
                className="text-xs"
              >
                {camera.status === "online" ? "온라인" : "오프라인"}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">{camera.location}</p>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="relative aspect-video bg-muted rounded-lg overflow-hidden">
              {camera.status === "online" ? (
                <ImageWithFallback
                  src={camera.imageUrl}
                  alt={camera.name}
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center bg-muted">
                  <Camera className="h-8 w-8 text-muted-foreground" />
                </div>
              )}
              
              {/* 실시간 표시 */}
              {camera.status === "online" && playingCameras.has(camera.id) && (
                <div className="absolute top-2 left-2">
                  <Badge variant="destructive" className="text-xs">
                    <div className="w-2 h-2 bg-white rounded-full mr-1 animate-pulse" />
                    LIVE
                  </Badge>
                </div>
              )}

              {/* 컨트롤 버튼들 */}
              <div className="absolute bottom-2 right-2 flex gap-1">
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-6 w-6 p-0"
                  onClick={() => toggleCamera(camera.id)}
                  disabled={camera.status === "offline"}
                >
                  {playingCameras.has(camera.id) ? (
                    <Pause className="h-3 w-3" />
                  ) : (
                    <Play className="h-3 w-3" />
                  )}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-6 w-6 p-0"
                  disabled={camera.status === "offline"}
                >
                  <Maximize className="h-3 w-3" />
                </Button>
              </div>
            </div>

            {/* AI 감지 결과 */}
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1">
                  <Users className="h-3 w-3 text-blue-500" />
                  <span>{camera.detections.people}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Car className="h-3 w-3 text-green-500" />
                  <span>{camera.detections.vehicles}</span>
                </div>
                {camera.detections.alerts > 0 && (
                  <div className="flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3 text-red-500" />
                    <span>{camera.detections.alerts}</span>
                  </div>
                )}
              </div>
              <span className="text-muted-foreground">
                {new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}