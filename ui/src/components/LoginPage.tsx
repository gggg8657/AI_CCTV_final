import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Shield } from "lucide-react";
import { useAuth } from "../context/AuthContext";
import { auth as authApi, setTokens } from "../lib/api";

export function LoginPage() {
  const { login } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "register") {
        await authApi.register(username, email, password);
      }
      await login(username, password);
    } catch (err: any) {
      setError(err.message || "로그인에 실패했습니다");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <Card className="w-full" style={{ maxWidth: 400 }}>
        <CardHeader className="text-center">
          <div className="mx-auto flex items-center justify-center w-12 h-12 bg-primary rounded-xl mb-4">
            <Shield className="h-7 w-7 text-primary-foreground" />
          </div>
          <CardTitle className="text-2xl">AI CCTV</CardTitle>
          <p className="text-sm text-muted-foreground">보안 모니터링 시스템</p>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label>사용자명</Label>
              <Input value={username} onChange={(e) => setUsername(e.target.value)} required autoFocus />
            </div>
            {mode === "register" && (
              <div className="space-y-2">
                <Label>이메일</Label>
                <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
              </div>
            )}
            <div className="space-y-2">
              <Label>비밀번호</Label>
              <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
            </div>
            {error && <p className="text-sm text-red-500">{error}</p>}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "처리 중..." : mode === "login" ? "로그인" : "가입 및 로그인"}
            </Button>
            <p className="text-center text-xs text-muted-foreground">
              {mode === "login" ? (
                <>
                  계정이 없으신가요?{" "}
                  <button type="button" className="underline" onClick={() => setMode("register")}>
                    회원가입
                  </button>
                </>
              ) : (
                <>
                  이미 계정이 있으신가요?{" "}
                  <button type="button" className="underline" onClick={() => setMode("login")}>
                    로그인
                  </button>
                </>
              )}
            </p>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
