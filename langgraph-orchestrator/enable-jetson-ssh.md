# Enable SSH Access on Jetson Node

The Jetson node (192.168.1.177) currently has SSH port 22 filtered/blocked. Here's how to enable SSH access:

## Steps to Enable SSH on Jetson Node

### 1. Physical/Direct Access to Jetson
Connect directly to the Jetson node (monitor, keyboard, or existing terminal session).

### 2. Install and Enable SSH Service
```bash
# Update package list
sudo apt update

# Install OpenSSH server if not installed
sudo apt install openssh-server -y

# Enable SSH service
sudo systemctl enable ssh
sudo systemctl start ssh

# Check SSH service status
sudo systemctl status ssh
```

### 3. Configure Firewall (if UFW is active)
```bash
# Check if UFW is active
sudo ufw status

# If UFW is active, allow SSH
sudo ufw allow ssh
sudo ufw allow 22/tcp

# Reload firewall rules
sudo ufw reload
```

### 4. Alternative: Configure iptables (if using iptables)
```bash
# Check current iptables rules
sudo iptables -L

# Allow SSH connections
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT

# Save iptables rules (Ubuntu/Debian)
sudo iptables-save > /etc/iptables/rules.v4
```

### 5. Test SSH Access from CPU Coordinator
From the CPU coordinator (this machine), test SSH:
```bash
# Test SSH connection
ssh sanzad@192.168.1.177

# If successful, exit and copy SSH key
ssh-copy-id sanzad@192.168.1.177
```

## Quick Commands for Jetson

Run these commands **on the Jetson node** (192.168.1.177):

```bash
# Enable SSH and configure firewall
sudo systemctl enable ssh --now
sudo ufw allow ssh || sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo systemctl status ssh
```

## Verification

After enabling SSH on the Jetson, run this from the CPU coordinator:
```bash
# Test SSH connection
ssh -o ConnectTimeout=5 sanzad@192.168.1.177 'echo "SSH is working!"'

# If working, copy SSH key
ssh-copy-id sanzad@192.168.1.177

# Verify passwordless access
ssh -o BatchMode=yes sanzad@192.168.1.177 'echo "Passwordless SSH is working!"'
```

## Troubleshooting

1. **Check SSH service**: `sudo systemctl status ssh`
2. **Check firewall**: `sudo ufw status` or `sudo iptables -L`
3. **Check SSH config**: `sudo nano /etc/ssh/sshd_config`
4. **Restart SSH**: `sudo systemctl restart ssh`

Once SSH is enabled, re-run the SSH setup script:
```bash
./setup-ssh-keys.sh
```
