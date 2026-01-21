import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Progress } from "./ui/progress";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from "recharts";
import { 
  Search, 
  Brain, 
  TrendingUp, 
  AlertTriangle, 
  Users, 
  Car,
  Clock,
  Target,
  Zap,
  Filter
} from "lucide-react";

const hourlyPatterns = [
  { hour: "00:00", normal: 2, unusual: 0, threat: 0 },
  { hour: "02:00", normal: 1, unusual: 0, threat: 0 },
  { hour: "04:00", normal: 0, unusual: 1, threat: 0 },
  { hour: "06:00", normal: 3, unusual: 0, threat: 0 },
  { hour: "08:00", normal: 12, unusual: 1, threat: 0 },
  { hour: "10:00", normal: 18, unusual: 2, threat: 0 },
  { hour: "12:00", normal: 35, unusual: 3, threat: 1 },
  { hour: "14:00", normal: 28, unusual: 2, threat: 0 },
  { hour: "16:00", normal: 22, unusual: 1, threat: 0 },
  { hour: "18:00", normal: 15, unusual: 1, threat: 0 },
  { hour: "20:00", normal: 8, unusual: 0, threat: 0 },
  { hour: "22:00", normal: 4, unusual: 0, threat: 0 }
];

const behaviorAnalysis = [
  { type: "정상 행동", count: 1847, percentage: 92.4 },
  { type: "의심 행동", count: 124, percentage: 6.2 },
  { type: "위험 행동", count: 28, percentage: 1.4 }
];

const anomalyTypes = [
  { name: "정상", value: 1847, color: "#22c55e" },
  { name: "의심", value: 124, color: "#f59e0b" },
  { name: "위험", value: 28, color: "#ef4444" }
];

const recentAnomalies = [
  {
    id: "1",
    timestamp: "14:23",
    type: "의심 행동",
    description: "주차장에서 장시간 배회",
    camera: "주차장 A구역",
    confidence: 78,
    severity: "medium"
  },
  {
    id: "2",
    timestamp: "13:45",
    type: "시스템 이상",
    description: "카메라 연결 끊어짐",
    camera: "비상출구",
    confidence: 100,
    severity: "high"
  },
  {
    id: "3",
    timestamp: "12:15",
    type: "정상 벗어남",
    description: "비정상 시간대 출입",
    camera: "정문 출입구",
    confidence: 65,
    severity: "low"
  },
  {
    id: "4",
    timestamp: "11:30",
    type: "의심 행동",
    description: "금지구역 접근 시도",
    camera: "복도 2층",
    confidence: 85,
    severity: "high"
  }
];

export function AILogAnalysis() {
  const [analysisType, setAnalysisType] = useState("realtime");
  const [timeRange, setTimeRange] = useState("today");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);

  const runAIAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);

    const steps = [20, 40, 60, 80, 100];
    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 800));
      setAnalysisProgress(step);
    }

    setIsAnalyzing(false);
    setAnalysisProgress(0);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "high": return "text-red-600 bg-red-50 border-red-200";
      case "medium": return "text-orange-600 bg-orange-50 border-orange-200";
      case "low": return "text-yellow-600 bg-yellow-50 border-yellow-200";
      default: return "text-gray-600 bg-gray-50 border-gray-200";
    }
  };

  const getSeverityLabel = (severity: string) => {
    switch (severity) {
      case "high": return "높음";
      case "medium": return "보통";
      case "low": return "낮음";
      default: return "알 수 없음";
    }
  };

  return (
    <div className="space-y-6">
      {/* 분석 설정 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            AI 로그 분석
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">분석 유형</label>
              <Select value={analysisType} onValueChange={setAnalysisType}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="realtime">실시간 분석</SelectItem>
                  <SelectItem value="pattern">패턴 분석</SelectItem>
                  <SelectItem value="anomaly">이상 탐지</SelectItem>
                  <SelectItem value="behavior">행동 분석</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">분석 기간</label>
              <Select value={timeRange} onValueChange={setTimeRange}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="today">오늘</SelectItem>
                  <SelectItem value="week">최근 7일</SelectItem>
                  <SelectItem value="month">최근 30일</SelectItem>
                  <SelectItem value="custom">사용자 정의</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-end">
              <Button 
                onClick={runAIAnalysis}
                disabled={isAnalyzing}
                className="w-full"
              >
                {isAnalyzing ? (
                  <>
                    <Search className="h-4 w-4 mr-2 animate-spin" />
                    분석 중...
                  </>
                ) : (
                  <>
                    <Search className="h-4 w-4 mr-2" />
                    AI 분석 실행
                  </>
                )}
              </Button>
            </div>
          </div>

          {isAnalyzing && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">분석 진행률</span>
                <span>{analysisProgress}%</span>
              </div>
              <Progress value={analysisProgress} className="h-2" />
            </div>
          )}
        </CardContent>
      </Card>

      {/* 분석 결과 */}
      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <TrendingUp className="h-4 w-4" />
            개요
          </TabsTrigger>
          <TabsTrigger value="patterns" className="flex items-center gap-2">
            <Clock className="h-4 w-4" />
            패턴
          </TabsTrigger>
          <TabsTrigger value="anomalies" className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            이상 탐지
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            인사이트
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* 주요 지표 */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Users className="h-5 w-5 text-blue-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">분석된 이벤트</p>
                    <p className="text-2xl font-bold">1,999</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-green-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">정상 이벤트</p>
                    <p className="text-2xl font-bold">1,847</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-orange-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">의심 이벤트</p>
                    <p className="text-2xl font-bold">124</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-red-500" />
                  <div>
                    <p className="text-sm text-muted-foreground">위험 이벤트</p>
                    <p className="text-2xl font-bold">28</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* 행동 분석 차트 */}
          <Card>
            <CardHeader>
              <CardTitle>행동 패턴 분석</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col lg:flex-row gap-6">
                <div className="lg:w-1/2">
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={anomalyTypes}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}`}
                      >
                        {anomalyTypes.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="lg:w-1/2 space-y-4">
                  {behaviorAnalysis.map((item, index) => (
                    <div key={index} className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">{item.type}</span>
                        <span className="text-sm text-muted-foreground">
                          {item.count}건 ({item.percentage}%)
                        </span>
                      </div>
                      <Progress value={item.percentage} className="h-2" />
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="patterns" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>시간대별 활동 패턴</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={hourlyPatterns}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="normal" fill="#22c55e" name="정상" />
                  <Bar dataKey="unusual" fill="#f59e0b" name="의심" />
                  <Bar dataKey="threat" fill="#ef4444" name="위험" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="anomalies" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>최근 이상 탐지 결과</span>
                <Button variant="outline" size="sm">
                  <Filter className="h-4 w-4 mr-2" />
                  필터
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentAnomalies.map((anomaly) => (
                  <div
                    key={anomaly.id}
                    className={`p-4 rounded-lg border ${getSeverityColor(anomaly.severity)}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge 
                            variant="outline" 
                            className={`${getSeverityColor(anomaly.severity)} border-current`}
                          >
                            {getSeverityLabel(anomaly.severity)}
                          </Badge>
                          <span className="text-sm font-medium">{anomaly.type}</span>
                        </div>
                        <p className="text-sm mb-1">{anomaly.description}</p>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span>📍 {anomaly.camera}</span>
                          <span>🕐 {anomaly.timestamp}</span>
                          <span>🎯 신뢰도 {anomaly.confidence}%</span>
                        </div>
                      </div>
                      <Button variant="outline" size="sm">
                        상세 보기
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI 인사이트 및 권장사항</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">🔍 패턴 분석 결과</h4>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>• 점심시간대(12-13시) 활동이 가장 활발함</li>
                    <li>• 주차장 A구역에서 의심 활동이 빈번히 감지됨</li>
                    <li>• 야간시간대 비정상 출입이 증가하는 추세</li>
                  </ul>
                </div>

                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <h4 className="font-medium text-green-900 mb-2">✅ 보안 강화 권장사항</h4>
                  <ul className="text-sm text-green-800 space-y-1">
                    <li>• 주차장 A구역 조명 개선 및 추가 카메라 설치</li>
                    <li>• 야간 출입 시 추가 인증 절차 도입</li>
                    <li>• 비상출구 카메라 즉시 점검 및 수리</li>
                  </ul>
                </div>

                <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                  <h4 className="font-medium text-orange-900 mb-2">⚡ 시스템 최적화</h4>
                  <ul className="text-sm text-orange-800 space-y-1">
                    <li>• AI 감지 모델 업데이트로 정확도 2% 향상 가능</li>
                    <li>• 저장공간 최적화로 30일 추가 데이터 보관 가능</li>
                    <li>• 실시간 알림 임계값 조정으로 오탐 50% 감소 예상</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}