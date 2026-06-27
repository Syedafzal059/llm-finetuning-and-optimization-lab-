/**
 * PM2 event bus listener — sends crash alerts only for unexpected restarts.
 *
 * Skips alerts for:
 * - manual `pm2 restart` / `pm2 stop`
 * - first process start
 * - clean exit (code 0)
 */

const pm2 = require("pm2");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT_DIR = path.resolve(__dirname, "..");

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
    env[trimmed.slice(0, separatorIndex).trim()] =
      trimmed.slice(separatorIndex + 1).trim();
  }

  return env;
}

const rootEnv = loadEnvFile(path.join(ROOT_DIR, ".env"));
const workspace = process.env.WORKSPACE || rootEnv.WORKSPACE || ROOT_DIR;
const pythonInterpreter =
  process.env.PM2_PYTHON || rootEnv.PM2_PYTHON || "python";
const alertScript =
  process.env.PM2_CRASH_ALERT_SCRIPT ||
  path.join(ROOT_DIR, "scripts", "crash_alert.py");

function shouldAlert(packet) {
  if (!packet || !packet.process || !packet.process.name) {
    return false;
  }

  if (packet.process.name === "pm2-crash-listener") {
    return false;
  }

  if (packet.event !== "exit") {
    return false;
  }

  // PM2 sets `man: true` for operator-initiated restarts/stops.
  if (packet.man === true) {
    return false;
  }

  const exitCode = packet.process.exit_code;
  if (exitCode === 0) {
    return false;
  }

  return true;
}

function sendCrashAlert(processMeta) {
  const projectName = processMeta.name;
  const restartCount = processMeta.restart_time || 0;
  const errorLog = path.join(
    workspace,
    "projects",
    projectName,
    "logs",
    "pm2_err.log",
  );

  const child = spawn(
    pythonInterpreter,
    [alertScript, projectName, String(restartCount), errorLog],
    {
      cwd: ROOT_DIR,
      env: { ...process.env, ...rootEnv, WORKSPACE: workspace },
      stdio: "inherit",
    },
  );

  child.on("error", (error) => {
    console.error(`Failed to run crash alert script: ${error.message}`);
  });
}

pm2.connect((connectError) => {
  if (connectError) {
    console.error(connectError);
    process.exit(2);
  }

  pm2.launchBus((busError, bus) => {
    if (busError) {
      console.error(busError);
      pm2.disconnect();
      process.exit(2);
    }

    console.log("PM2 crash listener running");

    bus.on("process:event", (packet) => {
      if (!shouldAlert(packet)) {
        return;
      }

      console.log(
        `Crash detected for ${packet.process.name} ` +
          `(exit_code=${packet.process.exit_code})`,
      );
      sendCrashAlert(packet.process);
    });
  });
});
