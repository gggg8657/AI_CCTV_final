import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { ScrollArea } from "./ui/scroll-area";
import { Avatar } from "./ui/avatar";
import {
  Send,
  Bot,
  User,
  Loader2,
  Clock,
  Shield,
  AlertTriangle,
  FileText,
  BarChart3,
  Camera,
} from "lucide-react";
import { cameras, events, stats, health } from "../lib/api";

interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
}

const quickActions = [
  { id: "status", label: "시스템 상태 확인", icon: Shield },
  { id: "alerts", label: "최근 알림 분석", icon: AlertTriangle },
  { id: "report", label: "요약 보고서", icon: FileText },
  { id: "stats", label: "통계 요약", icon: BarChart3 },
  { id: "cameras", label: "카메라 점검", icon: Camera },
];

async function fetchStatus(): Promise<string> {
  try {
    const [hp, camList] = await Promise.all([health(), cameras.list()]);
    const active = camList.filter((c) => c.status === "active").length;
    const inactive = camList.filter((c) => c.status !== "active");
    const pct = camList.length > 0 ? Math.round((active / camList.length) * 100) : 0;
    const dummy = hp?.pipeline?.dummy_flags;
    let txt = `시스템 상태를 확인했습니다.\n\n`;
    txt += `**카메라**: ${active}/${camList.length}대 활성 (가동률 ${pct}%)\n`;
    if (dummy) txt += `**모델 모드**: VAD ${dummy.vad ? "Dummy" : "Real"} / VLM ${dummy.vlm ? "Dummy" : "Real"} / Agent ${dummy.agent ? "Dummy" : "Real"}\n`;
    if (inactive.length > 0) {
      txt += `\n**비활성 카메라**\n`;
      inactive.forEach((c) => { txt += `- ${c.name} (${c.source_type})\n`; });
    }
    return txt;
  } catch (e: any) {
    return `시스템 상태 조회 실패: ${e.message}`;
  }
}

async function fetchAlerts(): Promise<string> {
  try {
    const result = await events.list({ limit: "10" });
    if (result.items.length === 0) return "최근 이벤트가 없습니다.";
    const unacked = result.items.filter((e) => !e.acknowledged).length;
    let txt = `최근 이벤트 분석 (총 ${result.total}건)\n\n`;
    txt += `**미확인**: ${unacked}건\n\n`;
    txt += `**최근 이벤트**\n`;
    result.items.slice(0, 5).forEach((ev) => {
      const t = new Date(ev.timestamp).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
      txt += `- ${t} 카메라#${ev.camera_id} — ${ev.vlm_type || "이상탐지"} (VAD ${ev.vad_score.toFixed(2)})${ev.acknowledged ? "" : " ⚠️"}\n`;
    });
    return txt;
  } catch (e: any) {
    return `이벤트 조회 실패: ${e.message}`;
  }
}

async function fetchReport(): Promise<string> {
  try {
    const [camList, evResult, summary] = await Promise.all([
      cameras.list(),
      events.list({ limit: "100" }),
      stats.summary(1).catch(() => null),
    ]);
    const active = camList.filter((c) => c.status === "active").length;
    const now = new Date().toLocaleDateString("ko-KR");
    let txt = `**${now} 보안 현황 요약**\n\n`;
    txt += `**1. 카메라 현황**\n- 전체: ${camList.length}대 / 활성: ${active}대\n\n`;
    txt += `**2. 이벤트 현황**\n- 총 이벤트: ${evResult.total}건\n`;
    if (summary) {
      txt += `- 미확인: ${summary.unacknowledged}건\n`;
      txt += `- 평균 VAD: ${summary.avg_vad_score.toFixed(4)}\n`;
      const types = summary.vlm_type_distribution;
      if (Object.keys(types).length > 0) {
        txt += `\n**3. 이벤트 유형 분포**\n`;
        Object.entries(types).forEach(([t, c]) => { txt += `- ${t}: ${c}건\n`; });
      }
    }
    return txt;
  } catch (e: any) {
    return `보고서 생성 실패: ${e.message}`;
  }
}

async function fetchStats(): Promise<string> {
  try {
    const summary = await stats.summary(7);
    let txt = `최근 ${summary.period_days}일 통계 요약\n\n`;
    txt += `**총 이벤트**: ${summary.total_events}건\n`;
    txt += `**미확인**: ${summary.unacknowledged}건\n`;
    txt += `**평균 VAD 점수**: ${summary.avg_vad_score.toFixed(4)}\n`;
    const types = summary.vlm_type_distribution;
    if (Object.keys(types).length > 0) {
      txt += `\n**이벤트 유형 분포**\n`;
      Object.entries(types).forEach(([t, c]) => { txt += `- ${t}: ${c}건\n`; });
    } else {
      txt += `\n이벤트 유형 분류 데이터 없음`;
    }
    return txt;
  } catch (e: any) {
    return `통계 조회 실패: ${e.message}`;
  }
}

async function fetchCameras(): Promise<string> {
  try {
    const camList = await cameras.list();
    if (camList.length === 0) return "등록된 카메라가 없습니다.";
    const active = camList.filter((c) => c.status === "active");
    const inactive = camList.filter((c) => c.status !== "active");
    let txt = `카메라 점검 결과\n\n`;
    if (active.length > 0) {
      txt += `**활성 (${active.length}대)**\n`;
      active.forEach((c) => { txt += `- ${c.name} (${c.source_type}) — ${c.location || "위치 미설정"}\n`; });
    }
    if (inactive.length > 0) {
      txt += `\n**비활성 (${inactive.length}대)**\n`;
      inactive.forEach((c) => { txt += `- ${c.name} (${c.source_type})\n`; });
    }
    return txt;
  } catch (e: any) {
    return `카메라 조회 실패: ${e.message}`;
  }
}

const actionHandlers: Record<string, () => Promise<string>> = {
  status: fetchStatus,
  alerts: fetchAlerts,
  report: fetchReport,
  stats: fetchStats,
  cameras: fetchCameras,
};

export function AIAgentChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content: "안녕하세요! AI 보안 어시스턴트입니다.\n\n시스템 상태, 이벤트 분석, 카메라 점검 등을 실시간 API 데이터 기반으로 안내해드립니다. 아래 버튼을 클릭하거나 질문해주세요.",
      sender: "ai",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const addAIMessage = async (fetcher: () => Promise<string>) => {
    setIsTyping(true);
    try {
      const response = await fetcher();
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), content: response, sender: "ai", timestamp: new Date() },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: Date.now().toString(), content: "응답을 가져오는 중 오류가 발생했습니다.", sender: "ai", timestamp: new Date() },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSend = async (message: string) => {
    if (!message.trim()) return;
    setMessages((prev) => [
      ...prev,
      { id: Date.now().toString(), content: message, sender: "user", timestamp: new Date() },
    ]);
    setInput("");

    const lower = message.toLowerCase();
    let handler: (() => Promise<string>) | undefined;
    if (lower.includes("상태") || lower.includes("시스템")) handler = actionHandlers.status;
    else if (lower.includes("알림") || lower.includes("이벤트")) handler = actionHandlers.alerts;
    else if (lower.includes("보고서") || lower.includes("리포트") || lower.includes("요약")) handler = actionHandlers.report;
    else if (lower.includes("통계") || lower.includes("현황")) handler = actionHandlers.stats;
    else if (lower.includes("카메라") || lower.includes("점검")) handler = actionHandlers.cameras;

    if (handler) {
      await addAIMessage(handler);
    } else {
      await addAIMessage(async () =>
        "해당 질문을 처리할 수 없습니다. 아래 키워드를 포함해주세요:\n- **상태/시스템**: 시스템 상태 확인\n- **알림/이벤트**: 최근 이벤트 분석\n- **보고서/요약**: 보안 현황 보고서\n- **통계/현황**: 통계 요약\n- **카메라/점검**: 카메라 상태 점검"
      );
    }
  };

  const handleQuickAction = async (actionId: string) => {
    const action = quickActions.find((a) => a.id === actionId);
    if (!action) return;
    setMessages((prev) => [
      ...prev,
      { id: Date.now().toString(), content: action.label, sender: "user", timestamp: new Date() },
    ]);
    await addAIMessage(actionHandlers[actionId]);
  };

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages, isTyping]);

  return (
    <div className="flex flex-col h-[600px]">
      <Card className="flex-1 flex flex-col">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5 text-blue-500" />
            AI 보안 어시스턴트
            <Badge variant="secondary" className="ml-auto">실시간 API</Badge>
          </CardTitle>
        </CardHeader>

        <CardContent className="flex-1 flex flex-col space-y-4">
          <ScrollArea className="flex-1 pr-4" ref={scrollAreaRef}>
            <div className="space-y-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${message.sender === "user" ? "flex-row-reverse" : "flex-row"}`}
                >
                  <Avatar className={`w-8 h-8 ${message.sender === "user" ? "bg-primary" : "bg-blue-500"}`}>
                    {message.sender === "user" ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </Avatar>
                  <div className={`flex-1 max-w-[80%] ${message.sender === "user" ? "text-right" : "text-left"}`}>
                    <div
                      className={`rounded-lg p-3 whitespace-pre-line ${
                        message.sender === "user" ? "bg-primary text-primary-foreground ml-auto" : "bg-muted"
                      }`}
                    >
                      {message.content}
                    </div>
                    <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {message.timestamp.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" })}
                    </div>
                  </div>
                </div>
              ))}

              {isTyping && (
                <div className="flex gap-3">
                  <Avatar className="w-8 h-8 bg-blue-500">
                    <Bot className="h-4 w-4 text-white" />
                  </Avatar>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="flex items-center gap-1">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">API 데이터 조회 중...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">빠른 명령어:</p>
            <div className="flex flex-wrap gap-2">
              {quickActions.map((action) => {
                const Icon = action.icon;
                return (
                  <Button
                    key={action.id}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickAction(action.id)}
                    disabled={isTyping}
                    className="flex items-center gap-1"
                  >
                    <Icon className="h-3 w-3" />
                    {action.label}
                  </Button>
                );
              })}
            </div>
          </div>

          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="질문하세요 (상태, 알림, 통계, 카메라, 보고서 ...)"
              onKeyPress={(e) => e.key === "Enter" && handleSend(input)}
              disabled={isTyping}
              className="flex-1"
            />
            <Button onClick={() => handleSend(input)} disabled={isTyping || !input.trim()} className="shrink-0">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
