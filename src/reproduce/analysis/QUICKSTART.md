# Analysis Server - Quick Start Guide

Get your experiment analysis server running in 5 minutes!

## 🚀 Installation

### 1. Install Server Dependencies

```bash
# Install server dependency group (watchdog + streamlit)
uv sync --group server
```

### 2. Create Directories

```bash
mkdir -p uploads datalake logs
```

## 🎯 Running Locally (Development)

### Start File Watcher

Terminal 1:
```bash
./analysis/watcher/run_watcher.sh
```

This watches `uploads/` for new `.tar.gz` files and auto-extracts them to `datalake/`.

### Start Dashboard

Terminal 2:
```bash
./analysis/ui/run_dashboard.sh
```

Access at: **http://localhost:8501**

### Test It

```bash
# Create a test experiment tar.gz
cd datalake
tar -czf test_exp.tar.gz env_lambda_sweep_mes_10000/
mv test_exp.tar.gz ../uploads/

# Watch the watcher terminal - should auto-extract
# Refresh dashboard to see it
```

## 🖥️ Deploying to Server

### Quick Deploy (Docker)

```bash
# On your server
cd RL4PSJoint
docker-compose up -d

# View logs
docker-compose logs -f

# Access at http://your-server:8501
```

### Production Deploy (systemd)

See [SERVER_SETUP.md](SERVER_SETUP.md) for detailed systemd setup.

Quick version:
```bash
# Edit service files with your paths
nano analysis/watcher/experiment-watcher.service
nano analysis/ui/streamlit-dashboard.service

# Install and start
sudo cp analysis/watcher/experiment-watcher.service /etc/systemd/system/
sudo cp analysis/ui/streamlit-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now experiment-watcher streamlit-dashboard
```

## 📤 Uploading Experiments

### From Local Machine

```bash
# Create tar.gz
cd datalake
tar -czf my_experiment.tar.gz my_experiment/

# Upload to server
scp my_experiment.tar.gz your-server:~/RL4PSJoint/uploads/

# Watcher auto-extracts, dashboard refreshes
```

### Automated Sync (Optional)

Add to your experiment runner:
```bash
# After experiment completes
tar -czf "${EXPERIMENT_NAME}.tar.gz" "datalake/${EXPERIMENT_NAME}"
rsync -av "${EXPERIMENT_NAME}.tar.gz" server:~/RL4PSJoint/uploads/
```

## 🎨 Using the Dashboard

### Single Experiment
1. Tab: "📊 Single Experiment"
2. Select experiment from dropdown
3. Click "Load Experiment"
4. View summary, find best hyperparameters
5. Generate sensitivity plots

### Compare Experiments
1. Tab: "🔬 Compare Experiments"
2. Select 2+ experiments
3. Click "Load and Combine"
4. Generate comparison plots

### Export Results
- Download best hyperparameters as CSV
- Save plots as images
- Export experiment list

## 🔧 Common Commands

```bash
# Check watcher status
sudo systemctl status experiment-watcher

# View watcher logs
sudo journalctl -u experiment-watcher -f

# Restart dashboard
sudo systemctl restart streamlit-dashboard

# Manually process existing files
python -m analysis.watcher.service --watch-dir uploads --datalake-dir datalake

# Clean up processed archives
rm -rf uploads/processed/*
```

## 📡 Remote Access (Tailscale - Recommended)

```bash
# On server
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# On local machine
# Access dashboard at: http://server-name:8501
# No port forwarding needed!
```

## 🐛 Troubleshooting

**Dashboard not showing experiments?**
- Click "🔄 Refresh Experiments"
- Check `datalake/` has experiments
- Verify permissions

**Watcher not extracting?**
- Check logs: `sudo journalctl -u experiment-watcher -f`
- Verify tar.gz is valid: `tar -tzf uploads/file.tar.gz`
- Check disk space: `df -h`

**Can't access remotely?**
- Verify streamlit runs on 0.0.0.0: `ps aux | grep streamlit`
- Check firewall: `sudo ufw status`
- Use Tailscale for secure access

## 📚 Next Steps

- See [SERVER_SETUP.md](SERVER_SETUP.md) for production deployment
- See [analysis/README.md](README.md) for Python API usage
- See [plt-notebooks/](../plt-notebooks/) for example notebooks

## 💡 Tips

1. **Performance**: Dashboard caches experiments for 5 minutes
2. **Disk Space**: Archives are saved to `uploads/processed/`
3. **Security**: Use Tailscale instead of exposing port 8501
4. **Monitoring**: Set up log rotation for `logs/`
5. **Backup**: Regularly backup `datalake/` directory

Happy experimenting! 🧪
