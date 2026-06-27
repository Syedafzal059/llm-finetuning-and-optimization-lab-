const fs = require("fs");
const path = require("path");

function loadEnvFile(filePath) {
  const env = {};
  if (!fs.existsSync(filePath)) {
    return env;
  }

  for (const line of fs.readFileSync(filePath, "utf8").split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }
    const separatorIndex = trimmed.indexOf("=");
    if (separatorIndex === -1) {
      continue;
    }
    const key = trimmed.slice(0, separatorIndex).trim();
    const value = trimmed.slice(separatorIndex + 1).trim();
    env[key] = value;
  }

  return env;
}

const rootDir = __dirname;
const rootEnv = loadEnvFile(path.join(rootDir, ".env"));
const workspace =
  process.env.WORKSPACE || rootEnv.WORKSPACE || rootDir;
const cudaDevice =
  process.env.CUDA_VISIBLE_DEVICES ||
  rootEnv.CUDA_VISIBLE_DEVICES ||
  "0";
const crashAlertScript = path.join(rootDir, "scripts", "crash_alert.py");
const pythonInterpreter =
  process.env.PM2_PYTHON || rootEnv.PM2_PYTHON || "python";

function buildApp(name, project, port) {
  const logDir = path.join(workspace, "projects", project, "logs");
  return {
    name,
    script: "run.py",
    interpreter: pythonInterpreter,
    args: `--project ${project} --mode serve --port ${port}`,
    cwd: workspace,
    watch: false,
    autorestart: true,
    max_restarts: 10,
    min_uptime: "10s",
    restart_delay: 5000,
    max_memory_restart: "20G",
    output: path.join(logDir, "pm2_out.log"),
    error: path.join(logDir, "pm2_err.log"),
    log_date_format: "YYYY-MM-DD HH:mm:ss",
    merge_logs: true,
    post_update: ["pip install -r requirements.txt"],
    env: {
      PYTHONUNBUFFERED: "1",
      CUDA_VISIBLE_DEVICES: cudaDevice,
      WORKSPACE: workspace,
      PM2_CRASH_ALERT_SCRIPT: crashAlertScript,
      ALERT_FROM_EMAIL: rootEnv.ALERT_FROM_EMAIL || process.env.ALERT_FROM_EMAIL || "",
      ALERT_TO_EMAIL: rootEnv.ALERT_TO_EMAIL || process.env.ALERT_TO_EMAIL || "",
      ALERT_EMAIL_PASSWORD:
        rootEnv.ALERT_EMAIL_PASSWORD || process.env.ALERT_EMAIL_PASSWORD || "",
    },
  };
}

module.exports = {
  apps: [
    buildApp("clinical-notes", "clinical-notes", 8001),
    buildApp("medical-coding", "medical-coding", 8002),
    buildApp("patient-support", "patient-support", 8003),
  ],
};
