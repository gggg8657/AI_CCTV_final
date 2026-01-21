import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { AIAgentChat } from "./AIAgentChat";
import { AIReportGenerator } from "./AIReportGenerator";
import { AILogAnalysis } from "./AILogAnalysis";
import { Bot, FileText, Search, MessageCircle } from "lucide-react";

export function AIAgentPanel() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">AI 어시스턴트</h1>
        <p className="text-muted-foreground">
          AI 기반 보안 분석, 보고서 생성, 대화형 지원 서비스
        </p>
      </div>

      <Tabs defaultValue="chat" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="chat" className="flex items-center gap-2">
            <MessageCircle className="h-4 w-4" />
            AI 채팅
          </TabsTrigger>
          <TabsTrigger value="reports" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            보고서 생성
          </TabsTrigger>
          <TabsTrigger value="analysis" className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            로그 분석
          </TabsTrigger>
        </TabsList>

        <TabsContent value="chat" className="mt-6">
          <AIAgentChat />
        </TabsContent>

        <TabsContent value="reports" className="mt-6">
          <AIReportGenerator />
        </TabsContent>

        <TabsContent value="analysis" className="mt-6">
          <AILogAnalysis />
        </TabsContent>
      </Tabs>
    </div>
  );
}