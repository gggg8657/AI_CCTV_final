import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Label } from "./ui/label";
import { Checkbox } from "./ui/checkbox";
import { Progress } from "./ui/progress";
import {
  FileText,
  Download,
  Loader2,
  Calendar,
  Users,
  AlertTriangle,
  BarChart3,
  Clock,
  CheckCircle,
} from "lucide-react";
import { cameras, events, stats } from "../lib/api";

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  duration: string;
  icon: React.ComponentType<any>;
}

interface GeneratedReport {
  id: string;
  name: string;
  type: string;
  generatedAt: Date;
  status: "completed" | "generating" | "failed";
  downloadUrl?: string;
}

const reportTemplates: ReportTemplate[] = [
  {
    id: "daily",
    name: "일일 보안 현황",
    description: "오늘 하루 전체 보안 활동과 이벤트 요약",
    duration: "~2분",
    icon: Calendar
  },
  {
    id: "weekly",
    name: "주간 통계 분석",
    description: "최근 7일간 패턴 분석 및 트렌드",
    duration: "~5분",
    icon: BarChart3
  },
  {
    id: "incident",
    name: "사건 상세 보고서",
    description: "특정 알림이나 이벤트에 대한 심층 분석",
    duration: "~3분",
    icon: AlertTriangle
  },
  {
    id: "traffic",
    name: "출입 통계 리포트",
    description: "인원 및 차량 출입 패턴 분석",
    duration: "~4분",
    icon: Users
  }
];


export function AIReportGenerator() {
  const [selectedTemplate, setSelectedTemplate] = useState("");
  const [selectedPeriod, setSelectedPeriod] = useState("today");
  const [selectedSections, setSelectedSections] = useState<string[]>([
    "summary", "detection", "alerts", "recommendations"
  ]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [reports, setReports] = useState<GeneratedReport[]>([]);

  const reportSections = [
    { id: "summary", label: "전체 요약", description: "핵심 지표와 현황 개요" },
    { id: "detection", label: "AI 감지 분석", description: "사람/차량 감지 통계" },
    { id: "alerts", label: "알림 및 이벤트", description: "보안 이벤트 상세 내역" },
    { id: "traffic", label: "출입 통계", description: "시간대별 출입 패턴" },
    { id: "cameras", label: "카메라 상태", description: "카메라 동작 현황" },
    { id: "recommendations", label: "개선 권장사항", description: "AI가 제안하는 보안 강화 방안" }
  ];

  const handleSectionToggle = (sectionId: string) => {
    setSelectedSections(prev =>
      prev.includes(sectionId)
        ? prev.filter(id => id !== sectionId)
        : [...prev, sectionId]
    );
  };

  const generateReport = async () => {
    setIsGenerating(true);
    setGenerationProgress(0);

    const periodDays: Record<string, number> = { today: 1, yesterday: 1, week: 7, month: 30, custom: 7 };
    const days = periodDays[selectedPeriod] ?? 7;

    try {
      setGenerationProgress(20);
      const [camList, evResult, summary] = await Promise.all([
        cameras.list(),
        events.list({ limit: "200" }),
        stats.summary(days).catch(() => null),
      ]);
      setGenerationProgress(60);

      const template = reportTemplates.find((t) => t.id === selectedTemplate);
      const active = camList.filter((c) => c.status === "active").length;
      const now = new Date();
      let content = `# ${template?.name} — ${now.toLocaleDateString("ko-KR")}\n\n`;

      if (selectedSections.includes("summary")) {
        content += `## 전체 요약\n- 카메라: ${active}/${camList.length}대 활성\n- 총 이벤트: ${evResult.total}건\n`;
        if (summary) content += `- 미확인: ${summary.unacknowledged}건\n- 평균 VAD: ${summary.avg_vad_score.toFixed(4)}\n`;
        content += "\n";
      }
      if (selectedSections.includes("alerts") && evResult.items.length > 0) {
        content += `## 이벤트 내역 (최근 ${Math.min(evResult.items.length, 10)}건)\n`;
        evResult.items.slice(0, 10).forEach((ev) => {
          const t = new Date(ev.timestamp).toLocaleString("ko-KR");
          content += `- ${t} | 카메라#${ev.camera_id} | ${ev.vlm_type || "이상탐지"} | VAD ${ev.vad_score.toFixed(2)}\n`;
        });
        content += "\n";
      }
      if (selectedSections.includes("cameras")) {
        content += `## 카메라 상태\n`;
        camList.forEach((c) => {
          content += `- ${c.name}: ${c.status === "active" ? "활성" : "비활성"} (${c.source_type})\n`;
        });
        content += "\n";
      }
      setGenerationProgress(90);

      const blob = new Blob([content], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);

      const newReport: GeneratedReport = {
        id: Date.now().toString(),
        name: `${template?.name} - ${now.toLocaleDateString("ko-KR")}`,
        type: selectedTemplate,
        generatedAt: now,
        status: "completed",
        downloadUrl: url,
      };

      setReports((prev) => [newReport, ...prev]);
      setGenerationProgress(100);
    } catch {
      const failedReport: GeneratedReport = {
        id: Date.now().toString(),
        name: `보고서 생성 실패`,
        type: selectedTemplate,
        generatedAt: new Date(),
        status: "failed",
      };
      setReports((prev) => [failedReport, ...prev]);
    } finally {
      setIsGenerating(false);
      setGenerationProgress(0);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed": return "text-green-600";
      case "generating": return "text-blue-600";
      case "failed": return "text-red-600";
      default: return "text-gray-600";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed": return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "generating": return <Loader2 className="h-4 w-4 text-blue-600 animate-spin" />;
      case "failed": return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default: return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* 보고서 생성 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            AI 보고서 생성
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* 템플릿 선택 */}
          <div className="space-y-4">
            <Label>보고서 유형 선택</Label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {reportTemplates.map((template) => {
                const Icon = template.icon;
                return (
                  <Card
                    key={template.id}
                    className={`cursor-pointer transition-all ${
                      selectedTemplate === template.id 
                        ? "ring-2 ring-primary bg-primary/5" 
                        : "hover:bg-muted/50"
                    }`}
                    onClick={() => setSelectedTemplate(template.id)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <Icon className="h-4 w-4 text-primary" />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h4 className="font-medium">{template.name}</h4>
                          <p className="text-sm text-muted-foreground mt-1">
                            {template.description}
                          </p>
                          <Badge variant="outline" className="mt-2">
                            생성 시간 {template.duration}
                          </Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>

          {/* 기간 선택 */}
          <div className="space-y-2">
            <Label>분석 기간</Label>
            <Select value={selectedPeriod} onValueChange={setSelectedPeriod}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="today">오늘</SelectItem>
                <SelectItem value="yesterday">어제</SelectItem>
                <SelectItem value="week">최근 7일</SelectItem>
                <SelectItem value="month">최근 30일</SelectItem>
                <SelectItem value="custom">사용자 정의</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* 포함 섹션 선택 */}
          <div className="space-y-3">
            <Label>포함할 섹션</Label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {reportSections.map((section) => (
                <div key={section.id} className="flex items-start space-x-2">
                  <Checkbox
                    id={section.id}
                    checked={selectedSections.includes(section.id)}
                    onCheckedChange={() => handleSectionToggle(section.id)}
                  />
                  <div className="grid gap-1.5 leading-none">
                    <label
                      htmlFor={section.id}
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      {section.label}
                    </label>
                    <p className="text-xs text-muted-foreground">
                      {section.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 생성 버튼 */}
          <div className="space-y-3">
            <Button
              onClick={generateReport}
              disabled={!selectedTemplate || isGenerating}
              className="w-full"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  보고서 생성 중...
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4 mr-2" />
                  AI 보고서 생성하기
                </>
              )}
            </Button>

            {/* 진행률 표시 */}
            {isGenerating && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">
                    {generationProgress < 100 ? "생성 진행률" : "완료!"}
                  </span>
                  <span>{generationProgress}%</span>
                </div>
                <Progress value={generationProgress} className="h-2" />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* 생성된 보고서 목록 */}
      <Card>
        <CardHeader>
          <CardTitle>생성된 보고서</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {reports.map((report) => (
              <div
                key={report.id}
                className="flex items-center justify-between p-3 border rounded-lg"
              >
                <div className="flex items-center gap-3">
                  {getStatusIcon(report.status)}
                  <div>
                    <p className="font-medium">{report.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {report.generatedAt.toLocaleString('ko-KR')}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={getStatusColor(report.status)}>
                    {report.status === "completed" ? "완료" : 
                     report.status === "generating" ? "생성중" : "실패"}
                  </Badge>
                  {report.status === "completed" && report.downloadUrl && (
                    <a href={report.downloadUrl} download={`${report.name}.md`}>
                      <Button variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-1" />
                        다운로드
                      </Button>
                    </a>
                  )}
                </div>
              </div>
            ))}
            
            {reports.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>아직 생성된 보고서가 없습니다.</p>
                <p className="text-sm">위에서 새 보고서를 생성해보세요.</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}