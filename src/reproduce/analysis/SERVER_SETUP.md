# Analysis Server Setup Guide

Complete guide for deploying the RL experiment analysis server on a remote machine.

## Overview

The analysis server consists of two components:
1. **File Watcher**: Monitors `uploads/` for new `.tar.gz` files and auto-extracts to `datalake/`
2. **Streamlit Dashboard**: Web UI for exploring experiments at `http://server:8501`

## Quick Start

### Option 1: Docker (Recommended for simplicity)

```bash
# On your server
cd RL4PSJoint

# Install Docker and Docker Compose if needed
# https://docs.docker.com/engine/install/

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Access dashboard at http://your-server:8501
```

### Option 2: Direct Installation

```bash
# On your server
cd RL4PSJoint

# Install dependencies
uv pip install watchdog streamlit

# Start watcher (in terminal 1)
./analysis/watcher/run_watcher.sh

# Start dashboard (in terminal 2)
./analysis/ui/run_dashboard.sh

# Access at http://your-server:8501
```

### Option 3: Systemd Services (Best for always-on server)

See detailed instructions below.

## Detailed Setup

### 1. Server Preparation

```bash
# SSH to your server
ssh your-server

# Clone/sync your project
cd /path/to/RL4PSJoint

# Create necessary directories
mkdir -p uploads datalake logs

# Set permissions
chmod +x analysis/watcher/run_watcher.sh
chmod +x analysis/ui/run_dashboard.sh
```

### 2. Install Dependencies

```bash
# Install Python dependencies
uv pip install watchdog streamlit

# Or with regular pip
pip install watchdog streamlit
```

### 3. Test Services Locally

```bash
# Test watcher
python -m analysis.watcher.service --watch-dir uploads --datalake-dir datalake

# In another terminal, test dashboard
streamlit run analysis/ui/streamlit_app.py

# Test by dropping a tar.gz file in uploads/
# Should see it auto-extract to datalake/
```

### 4. Setup Systemd Services (Optional but recommended)

**Edit service files with your paths:**

```bash
# Edit watcher service
nano analysis/watcher/experiment-watcher.service

# Update these lines:
#   User=YOUR_USERNAME
#   WorkingDirectory=/FULL/PATH/TO/RL4PSJoint
#   Environment="PATH=/FULL/PATH/TO/.venv/bin:/usr/local/bin:/usr/bin"
#   ExecStart=/FULL/PATH/TO/RL4PSJoint/analysis/watcher/run_watcher.sh

# Edit dashboard service
nano analysis/ui/streamlit-dashboard.service

# Update same lines as above
```

**Install services:**

```bash
# Copy service files
sudo cp analysis/watcher/experiment-watcher.service /etc/systemd/system/
sudo cp analysis/ui/streamlit-dashboard.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable experiment-watcher
sudo systemctl enable streamlit-dashboard

# Start services
sudo systemctl start experiment-watcher
sudo systemctl start streamlit-dashboard

# Check status
sudo systemctl status experiment-watcher
sudo systemctl status streamlit-dashboard

# View logs
sudo journalctl -u experiment-watcher -f
sudo journalctl -u streamlit-dashboard -f
```

**Manage services:**

```bash
# Stop services
sudo systemctl stop experiment-watcher
sudo systemctl stop streamlit-dashboard

# Restart services
sudo systemctl restart experiment-watcher
sudo systemctl restart streamlit-dashboard

# Disable services (don't start on boot)
sudo systemctl disable experiment-watcher
sudo systemctl disable streamlit-dashboard
```

### 5. Configure Firewall (if needed)

```bash
# Allow Streamlit port
sudo ufw allow 8501/tcp

# Or if using Tailscale (recommended for security)
# Dashboard will be accessible only on your tailnet
# No firewall changes needed!
```

### 6. Access Dashboard

```bash
# If using Tailscale
http://your-machine-name:8501

# Or with IP
http://192.168.x.x:8501

# Or if port forwarding
http://your-public-ip:8501
```

## Usage Workflow

### Upload New Experiments

**From your local machine:**

```bash
# Create tar.gz of experiment
cd datalake
tar -czf my_experiment.tar.gz my_experiment/

# Upload to server
scp my_experiment.tar.gz your-server:~/RL4PSJoint/uploads/

# Or with rsync
rsync -av my_experiment.tar.gz your-server:~/RL4PSJoint/uploads/
```

**Watcher will automatically:**
1. Detect new `.tar.gz` file
2. Extract to `datalake/my_experiment/`
3. Validate structure
4. Move tar.gz to `uploads/processed/`
5. Log everything

**View in dashboard:**
1. Open `http://your-server:8501`
2. Click "🔄 Refresh Experiments"
3. Select and explore!

### Explore Experiments

**Single Experiment:**
1. Tab: "📊 Single Experiment"
2. Select experiment
3. View summary, best hyperparameters
4. Generate sensitivity plots

**Compare Experiments:**
1. Tab: "🔬 Compare Experiments"
2. Select 2+ experiments
3. Click "Load and Combine"
4. Generate comparison plots

**All Experiments:**
1. Tab: "📋 Experiment List"
2. View all experiments
3. Download CSV

## Configuration

Edit `analysis/config.yaml`:

```yaml
watcher:
  watch_dir: "uploads"
  datalake_dir: "datalake"
  archive_dir: "uploads/processed"
  auto_archive: true

streamlit:
  port: 8501
  host: "0.0.0.0"
```

## Troubleshooting

### Watcher not detecting files

```bash
# Check watcher logs
sudo journalctl -u experiment-watcher -n 50

# Test manually
python -m analysis.watcher.service --watch-dir uploads --datalake-dir datalake

# Check permissions
ls -la uploads/
```

### Dashboard not loading data

```bash
# Check dashboard logs
sudo journalctl -u streamlit-dashboard -n 50

# Verify datalake directory
ls -la datalake/

# Test loading
python -c "from analysis import DataLake; lake = DataLake('datalake'); print(lake.list_experiments())"
```

### Can't access dashboard remotely

```bash
# Check if Streamlit is listening
sudo netstat -tulpn | grep 8501

# Check firewall
sudo ufw status

# Verify Streamlit is bound to 0.0.0.0 not 127.0.0.1
ps aux | grep streamlit
```

### Out of disk space

```bash
# Check disk usage
df -h

# Clean processed archives
rm -rf uploads/processed/*

# Compress old experiments
cd datalake
tar -czf old_experiments.tar.gz old_experiment_1/ old_experiment_2/
rm -rf old_experiment_1/ old_experiment_2/
```

## Monitoring

### Check service health

```bash
# Service status
sudo systemctl status experiment-watcher streamlit-dashboard

# Recent logs
sudo journalctl -u experiment-watcher -u streamlit-dashboard --since "1 hour ago"

# Follow logs live
sudo journalctl -u experiment-watcher -u streamlit-dashboard -f
```

### Disk usage

```bash
# Check sizes
du -sh uploads/ datalake/ logs/

# Monitor growth
watch -n 60 'du -sh uploads/ datalake/'
```

## Backup

### Backup datalake

```bash
# Full backup
tar -czf datalake_backup_$(date +%Y%m%d).tar.gz datalake/

# Sync to backup server
rsync -av --progress datalake/ backup-server:/backups/datalake/

# Backup to cloud (if using rclone)
rclone sync datalake/ remote:backups/datalake/
```

### Restore

```bash
# Extract backup
tar -xzf datalake_backup_20240101.tar.gz

# Or restore from sync
rsync -av backup-server:/backups/datalake/ datalake/
```

## Security

### Tailscale (Recommended)

```bash
# Install Tailscale on server
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# Dashboard now accessible only on your tailnet
# http://your-server-name:8501
```

### Nginx Reverse Proxy with Auth (Alternative)

```nginx
# /etc/nginx/sites-available/streamlit
server {
    listen 80;
    server_name your-domain.com;

    location / {
        auth_basic "RL Experiments";
        auth_basic_user_file /etc/nginx/.htpasswd;

        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Performance Tips

1. **Cache**: Streamlit auto-caches loaded experiments (5 min TTL)
2. **Large experiments**: Consider increasing cache TTL in `streamlit_app.py`
3. **Memory**: Monitor memory usage, restart services if needed
4. **Logs**: Rotate logs to prevent disk fill

```bash
# Add to crontab for log rotation
0 0 * * 0 find logs/ -name "*.log" -mtime +30 -delete
```

## Advanced: Custom Deployment

### PM2 (Node process manager)

```bash
# Install PM2
npm install -g pm2

# Start watcher
pm2 start analysis/watcher/run_watcher.sh --name watcher

# Start dashboard
pm2 start analysis/ui/run_dashboard.sh --name dashboard

# Save config
pm2 save
pm2 startup
```

### Kubernetes

See `kubernetes/` directory for manifests (coming soon).

## Support

Issues or questions? Check:
- Watcher logs: `sudo journalctl -u experiment-watcher -f`
- Dashboard logs: `sudo journalctl -u streamlit-dashboard -f`
- GitHub issues: https://github.com/anthropics/claude-code/issues
