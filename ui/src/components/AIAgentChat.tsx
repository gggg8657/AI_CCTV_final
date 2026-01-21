import { useState, useRef, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Badge } from "./ui/badge";
import { ScrollArea } from "./ui/scroll-area";
import { Avatar } from "./ui/avatar";
import { 
  MessageCircle, 
  Send, 
  Bot, 
  User, 
  Loader2, 
  Clock,
  Shield,
  AlertTriangle,
  FileText,
  BarChart3,
  Camera
} from "lucide-react";

interface Message {
  id: string;
  content: string;
  sender: "user" | "ai";
  timestamp: Date;
  type?: "text" | "analysis" | "report";
}

const quickActions = [
  { id: "status", label: "시스템 상태 확인", icon: Shield },
  { id: "alerts", label: "최근 알림 분석", icon: AlertTriangle },
  { id: "report", label: "일일 보고서 생성", icon: FileText },
  { id: "stats", label: "통계 요약", icon: BarChart3 },
  { id: "cameras", label: "카메라 점검", icon: Camera }
];

const mockResponses: Record<string, string> = {
  status: "현재 시스템 상태를 확인했습니다.\n\n✅ **전체 시스템 정상 운영 중**\n- 활성 카메라: 15/18대\n- 시스템 가동률: 95%\n- AI 감지 정확도: 98.2%\n\n⚠️ **주의사항**\n- 비상출구 카메라 오프라인 상태\n- 저장공간 사용률 78% (권장: 80% 이하)",
  alerts: "최근 24시간 알림을 분석했습니다.\n\n📊 **알림 통계**\n- 총 알림 수: 12건\n- 긴급 알림: 1건\n- 일반 알림: 11건\n\n🔍 **주요 이벤트**\n- 13:40 비상출구 카메라 연결 끊어짐\n- 12:30 주차장 A구역 차량 증가 (정상 범위)\n- 11:15 정문 출입 인원 급증 (점심시간 정상)",
  report: "일일 보고서를 생성하고 있습니다...\n\n📋 **2024년 9월 24일 보안 현황 보고서**\n\n**1. 전체 현황**\n- 총 방문자: 248명 (+12% 전일 대비)\n- 차량 출입: 156대 (-3% 전일 대비)\n- 보안 이벤트: 12건\n\n**2. AI 감지 현황**\n- 사람 감지: 98.5% 정확도\n- 차량 감지: 97.8% 정확도\n- 이상 행동 감지: 0건\n\n**3. 권장사항**\n- 비상출구 카메라 점검 필요\n- 주차장 조명 개선 검토\n\n📄 상세 보고서가 생성되었습니다.",
  stats: "금일 통계를 요약해드립니다.\n\n📈 **핵심 지표**\n- 피크 시간: 12:00-13:00 (45명 동시 감지)\n- 최다 출입 구역: 정문 출입구\n- 차량 점유율: 87% (주차장 A구역)\n\n📊 **시간대별 패턴**\n- 오전 9-11시: 출근 시간대 증가\n- 오후 12-13시: 점심시간 최대 활동\n- 오후 18시 이후: 활동량 감소\n\n💡 **인사이트**\n- 정상적인 업무 패턴 유지\n- 보안 위험도: 낮음",
  cameras: "전체 카메라 상태를 점검했습니다.\n\n🟢 **정상 카메라 (15대)**\n- 정문 출입구: 정상\n- 주차장 A구역: 정상\n- 복도 2층: 정상\n- 기타 12대: 모두 정상\n\n🔴 **문제 카메라 (1대)**\n- 비상출구: 연결 끊어짐 (13:40부터)\n\n⚡ **권장 조치**\n1. 비상출구 카메라 전원 및 네트워크 확인\n2. 정기 점검 일정: 매주 금요일 17:00\n3. 예비 카메라 설치 검토"
};

export function AIAgentChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      content: "안녕하세요! AI 보안 어시스턴트입니다. 🤖\n\n시스템 상태 확인, 보고서 생성, 로그 분석 등 무엇이든 도와드릴게요. 아래 버튼을 클릭하시거나 직접 질문해주세요!",
      sender: "ai",
      timestamp: new Date(),
      type: "text"
    }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const simulateTyping = async (response: string) => {
    setIsTyping(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const newMessage: Message = {
      id: Date.now().toString(),
      content: response,
      sender: "ai",
      timestamp: new Date(),
      type: "text"
    };
    
    setMessages(prev => [...prev, newMessage]);
    setIsTyping(false);
  };

  const handleSend = async (message: string) => {
    if (!message.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: message,
      sender: "user",
      timestamp: new Date(),
      type: "text"
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");

    // 간단한 키워드 매칭으로 응답 결정
    let response = "죄송합니다. 해당 요청을 처리할 수 없습니다. 다른 질문을 해주시거나 아래 버튼 중 하나를 선택해주세요.";
    
    const lowerMessage = message.toLowerCase();
    if (lowerMessage.includes("상태") || lowerMessage.includes("시스템")) {
      response = mockResponses.status;
    } else if (lowerMessage.includes("알림") || lowerMessage.includes("이벤트")) {
      response = mockResponses.alerts;
    } else if (lowerMessage.includes("보고서") || lowerMessage.includes("리포트")) {
      response = mockResponses.report;
    } else if (lowerMessage.includes("통계") || lowerMessage.includes("현황")) {
      response = mockResponses.stats;
    } else if (lowerMessage.includes("카메라") || lowerMessage.includes("점검")) {
      response = mockResponses.cameras;
    }

    await simulateTyping(response);
  };

  const handleQuickAction = async (actionId: string) => {
    const action = quickActions.find(a => a.id === actionId);
    if (!action) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: action.label,
      sender: "user",
      timestamp: new Date(),
      type: "text"
    };

    setMessages(prev => [...prev, userMessage]);
    await simulateTyping(mockResponses[actionId]);
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
            <Badge variant="secondary" className="ml-auto">온라인</Badge>
          </CardTitle>
        </CardHeader>
        
        <CardContent className="flex-1 flex flex-col space-y-4">
          {/* 메시지 영역 */}
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
                        message.sender === "user"
                          ? "bg-primary text-primary-foreground ml-auto"
                          : "bg-muted"
                      }`}
                    >
                      {message.content}
                    </div>
                    <div className="flex items-center gap-1 mt-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {message.timestamp.toLocaleTimeString('ko-KR', { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      })}
                    </div>
                  </div>
                </div>
              ))}
              
              {/* 타이핑 인디케이터 */}
              {isTyping && (
                <div className="flex gap-3">
                  <Avatar className="w-8 h-8 bg-blue-500">
                    <Bot className="h-4 w-4 text-white" />
                  </Avatar>
                  <div className="bg-muted rounded-lg p-3">
                    <div className="flex items-center gap-1">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">AI가 응답 중입니다...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>

          {/* 빠른 액션 버튼들 */}
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

          {/* 입력 영역 */}
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="AI 어시스턴트에게 질문하세요..."
              onKeyPress={(e) => e.key === "Enter" && handleSend(input)}
              disabled={isTyping}
              className="flex-1"
            />
            <Button 
              onClick={() => handleSend(input)}
              disabled={isTyping || !input.trim()}
              className="shrink-0"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}