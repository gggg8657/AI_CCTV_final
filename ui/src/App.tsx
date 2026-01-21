import { useState } from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarProvider,
  SidebarTrigger,
} from "./components/ui/sidebar";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { LiveCameraGrid } from "./components/LiveCameraGrid";
import { AIAnalysisPanel } from "./components/AIAnalysisPanel";
import { StatsDashboard } from "./components/StatsDashboard";
import { SettingsPanel } from "./components/SettingsPanel";
import { AIAgentPanel } from "./components/AIAgentPanel";
import {
  Camera,
  BarChart3,
  Brain,
  Settings,
  Shield,
  Bell,
  Home,
  LogOut,
  Bot,
} from "lucide-react";

const menuItems = [
  { id: "live", label: "실시간 모니터링", icon: Camera },
  { id: "analysis", label: "AI 분석", icon: Brain },
  { id: "agent", label: "AI 어시스턴트", icon: Bot },
  { id: "stats", label: "통계 및 리포트", icon: BarChart3 },
  { id: "settings", label: "설정", icon: Settings },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("live");
  const [activeAlerts] = useState(1);

  const renderContent = () => {
    switch (activeTab) {
      case "live":
        return (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">
                  실시간 모니터링
                </h1>
                <p className="text-muted-foreground">
                  모든 카메라의 실시간 영상을 확인하고 AI 분석
                  결과를 모니터링합니다
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Badge
                  variant="outline"
                  className="flex items-center gap-1"
                >
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  시스템 정상
                </Badge>
                {activeAlerts > 0 && (
                  <Badge
                    variant="destructive"
                    className="flex items-center gap-1"
                  >
                    <Bell className="h-3 w-3" />
                    {activeAlerts}개 알림
                  </Badge>
                )}
              </div>
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              <div className="xl:col-span-2">
                <LiveCameraGrid />
              </div>
              <div>
                <AIAnalysisPanel />
              </div>
            </div>
          </div>
        );
      case "analysis":
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold">AI 분석</h1>
              <p className="text-muted-foreground">
                인공지능이 분석한 실시간 감지 결과와 패턴을
                확인합니다
              </p>
            </div>
            <AIAnalysisPanel />
          </div>
        );
      case "agent":
        return <AIAgentPanel />;
      case "stats":
        return (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold">
                통계 및 리포트
              </h1>
              <p className="text-muted-foreground">
                시스템 사용량과 감지 패턴에 대한 상세한 통계를
                확인합니다
              </p>
            </div>
            <StatsDashboard />
          </div>
        );
      case "settings":
        return (
          <div>
            <SettingsPanel />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background">
        <Sidebar className="border-r">
          <SidebarHeader className="border-b p-4">
            <div className="flex items-center gap-2">
              <div className="flex items-center justify-center w-8 h-8 bg-primary rounded-lg">
                <Shield className="h-5 w-5 text-primary-foreground" />
              </div>
              <div>
                <h2 className="font-semibold">AI CCTV</h2>
                <p className="text-xs text-muted-foreground">
                  보안 모니터링 시스템
                </p>
              </div>
            </div>
          </SidebarHeader>

          <SidebarContent className="p-4">
            <SidebarMenu>
              {menuItems.map((item) => {
                const Icon = item.icon;
                return (
                  <SidebarMenuItem key={item.id}>
                    <SidebarMenuButton
                      onClick={() => setActiveTab(item.id)}
                      isActive={activeTab === item.id}
                      className="w-full justify-start"
                    >
                      <Icon className="h-4 w-4" />
                      <span>{item.label}</span>
                      {item.id === "analysis" &&
                        activeAlerts > 0 && (
                          <Badge
                            variant="destructive"
                            className="ml-auto text-xs"
                          >
                            {activeAlerts}
                          </Badge>
                        )}
                      {item.id === "agent" && (
                        <Badge
                          variant="secondary"
                          className="ml-auto text-xs"
                        >
                          AI
                        </Badge>
                      )}
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarContent>

          <div className="mt-auto p-4 border-t">
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Home className="h-4 w-4" />
                <span>본사 빌딩</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span>관리자: 김보안</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start text-muted-foreground"
              >
                <LogOut className="h-4 w-4 mr-2" />
                로그아웃
              </Button>
            </div>
          </div>
        </Sidebar>

        <div className="flex-1 flex flex-col">
          <header className="border-b p-4 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="flex items-center gap-4">
              <SidebarTrigger />
              <div className="flex items-center gap-2 ml-auto">
                <Button
                  variant="ghost"
                  size="sm"
                  className="relative"
                >
                  <Bell className="h-4 w-4" />
                  {activeAlerts > 0 && (
                    <Badge
                      variant="destructive"
                      className="absolute -top-1 -right-1 w-5 h-5 text-xs p-0 flex items-center justify-center"
                    >
                      {activeAlerts}
                    </Badge>
                  )}
                </Button>
                <div className="flex items-center gap-2 text-sm">
                  <div className="w-2 h-2 bg-green-500 rounded-full" />
                  <span className="text-muted-foreground">
                    {new Date().toLocaleString("ko-KR", {
                      month: "long",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>
              </div>
            </div>
          </header>

          <main className="flex-1 p-6 overflow-auto">
            {renderContent()}
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
}