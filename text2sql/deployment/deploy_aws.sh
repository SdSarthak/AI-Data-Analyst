#!/bin/bash
# AWS EC2 Deployment Script for Text-to-SQL LLM Application

set -e

echo "Starting Text-to-SQL LLM deployment..."

# Update system packages
sudo apt-get update
sudo apt-get upgrade -y

# Install Python and dependencies
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Create application directory
APP_DIR="/home/ubuntu/text-to-sql-llm"
sudo mkdir -p $APP_DIR
cd $APP_DIR

# Clone repository (replace with your repo URL)
# sudo git clone https://your-repo-url.git .

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file (user must fill in credentials)
if [ ! -f .env ]; then
    echo "Creating .env file - please fill in your credentials"
    cp .env.example .env
    echo "IMPORTANT: Edit .env file with your actual credentials before running the application"
fi

# Install Streamlit as a system service
sudo tee /etc/systemd/system/text-to-sql.service > /dev/null <<EOF
[Unit]
Description=Text-to-SQL LLM Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable text-to-sql
sudo systemctl start text-to-sql

# Install and configure Nginx as reverse proxy
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/text-to-sql > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable Nginx configuration
sudo ln -sf /etc/nginx/sites-available/text-to-sql /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo "Deployment completed successfully!"
echo "Application should be running at http://your-instance-ip"
echo "Don't forget to:"
echo "1. Configure your security group to allow inbound traffic on ports 80 and 443"
echo "2. Fill in the .env file with your credentials"
echo "3. Set up HTTPS with: sudo certbot --nginx -d your-domain.com"
