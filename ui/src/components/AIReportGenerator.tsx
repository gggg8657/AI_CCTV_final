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
  Car,
  AlertTriangle,
  BarChart3,
  Clock,
  CheckCircle
} from "lucide-react";

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

const mockReports: GeneratedReport[] = [
  {
    id: "1",
    name: "일일 보안 현황 - 9월 24일",
    type: "daily",
    generatedAt: new Date(Date.now() - 2 * 60 * 60 * 1000),
    status: "completed",
    downloadUrl: "#"
  },
  {
    id: "2",
    name: "주간 통계 분석 - 9월 3주차",
    type: "weekly",
    generatedAt: new Date(Date.now() - 24 * 60 * 60 * 1000),
    status: "completed",
    downloadUrl: "#"
  },
  {
    id: "3",
    name: "비상출구 카메라 오류 분석",
    type: "incident",
    generatedAt: new Date(Date.now() - 30 * 60 * 1000),
    status: "completed",
    downloadUrl: "#"
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
  const [reports, setReports] = useState<GeneratedReport[]>(mockReports);

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

  const simulateReportGeneration = async () => {
    setIsGenerating(true);
    setGenerationProgress(0);

    // 진행률 시뮬레이션
    const progressSteps = [15, 35, 60, 80, 95, 100];
    const stepLabels = [
      "데이터 수집 중...",
      "AI 분석 진행 중...",
      "패턴 분석 중...",
      "보고서 작성 중...",
      "검토 및 최적화 중...",
      "완료!"
    ];

    for (let i = 0; i < progressSteps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 800));
      setGenerationProgress(progressSteps[i]);
    }

    // 새 보고서 추가
    const template = reportTemplates.find(t => t.id === selectedTemplate);
    const newReport: GeneratedReport = {
      id: Date.now().toString(),
      name: `${template?.name} - ${new Date().toLocaleDateString('ko-KR')}`,
      type: selectedTemplate,
      generatedAt: new Date(),
      status: "completed",
      downloadUrl: "#"
    };

    setReports(prev => [newReport, ...prev]);
    setIsGenerating(false);
    setGenerationProgress(0);
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
              onClick={simulateReportGeneration}
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
                  {report.status === "completed" && (
                    <Button variant="outline" size="sm">
                      <Download className="h-4 w-4 mr-1" />
                      다운로드
                    </Button>
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