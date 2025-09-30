import docker
import toml
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any

class DockerImageBuilder:
    """Docker image builder"""
    
    def __init__(self, timeout=300):
        self.logger = logging.getLogger(__name__)
        self.client = docker.from_env(timeout=timeout)
        self.api_client = docker.APIClient(timeout=timeout)  # Add low-level API client
        self.base_path = Path(__file__).parent

        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration file"""
        config_path = self.base_path / "config.toml"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return toml.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            raise
    
    def _read_python_version(self, repo: str) -> str:
        """Read recommended Python version from project"""
        version_file = self.base_path / "swap" / self.config['files']['recommended_python_version']
        
        try:
            if version_file.exists():
                version = json.loads(version_file.read_text())
                self.logger.info(f"Read Python version from file: {version[repo]}")
                return version[repo]
            else:
                default_version = self.config['execution']['default_python_version']
                self.logger.info(f"Version file not found, using default version: {default_version}")
                return default_version
        except Exception as e:
            self.logger.warning(f"Failed to read Python version: {e}, using default version")
            return self.config['execution']['default_python_version']
    
    def _generate_dockerfile_content(self, python_version: str) -> str:
        """Generate Dockerfile content"""
        template = self.config['dockerfile']['template']
        
        # Replace placeholders in template
        dockerfile_content = template.format(
            base_image=f"python:{python_version}-bullseye",
            agent_prompt= "swap/trae-agent/trae_agent/prompt/agent_prompt.py"
        )
        
        # Add proxy environment variables to Dockerfile beginning
        proxy_lines = []
        if self.config.get('proxy', {}).get('enabled', False):
            proxy_config = self.config['proxy']
            if proxy_config.get('http_proxy'):
                proxy_lines.append(f"ARG HTTP_PROXY={proxy_config['http_proxy']}")
                proxy_lines.append(f"ARG http_proxy={proxy_config['http_proxy']}")
            if proxy_config.get('https_proxy'):
                proxy_lines.append(f"ARG HTTPS_PROXY={proxy_config['https_proxy']}")
                proxy_lines.append(f"ARG https_proxy={proxy_config['https_proxy']}")
        
        if proxy_lines:
            dockerfile_content = '\n'.join(proxy_lines) + '\n\n' + dockerfile_content
        
        return dockerfile_content
    
    def build_image(self, repo: str) -> str:
        """Build Docker image"""
        # Read Python version
        python_version = self._read_python_version(repo)
        
        # Generate image name
        image_name = f"codegen_{python_version}"
        
        # Check if image already exists
        try:
            existing_image = self.client.images.get(image_name)
            self.logger.info(f"Found existing image: {image_name}")
            return image_name
        except docker.errors.ImageNotFound:
            pass
        
        # Generate Dockerfile content
        dockerfile_content = self._generate_dockerfile_content(python_version)
        
        # Prepare build arguments
        buildargs = {}
        if self.config.get('proxy', {}).get('enabled', False):
            proxy_config = self.config['proxy']
            if proxy_config.get('http_proxy'):
                buildargs['HTTP_PROXY'] = proxy_config['http_proxy']
                buildargs['http_proxy'] = proxy_config['http_proxy']
            if proxy_config.get('https_proxy'):
                buildargs['HTTPS_PROXY'] = proxy_config['https_proxy']
                buildargs['https_proxy'] = proxy_config['https_proxy']
        
        # Create temporary Dockerfile in project root directory
        dockerfile_path = self.base_path / "Dockerfile.tmp"
        
        try:
            dockerfile_path.write_text(dockerfile_content)
            
            self.logger.info(f"Starting image build: {image_name} (Python {python_version})")
            if buildargs:
                self.logger.info(f"Using proxy settings: {list(buildargs.keys())}")
            
            # Use project root directory as build context
            for chunk in self.api_client.build(
                path=str(self.base_path),
                tag=image_name,
                rm=True,
                forcerm=True,
                dockerfile="Dockerfile.tmp",
                network_mode="host",
                buildargs=buildargs,
                decode=True
            ):
                if 'stream' in chunk:
                    log_line = chunk['stream'].strip()
                    if log_line:
                        print(log_line, flush=True)  # Print directly to console
                        self.logger.info(log_line)  # Also log to logger
            
            self.logger.info(f"Image build successful: {image_name}")
            return image_name
            
        except Exception as e:
            self.logger.error(f"Image build failed: {e}")
            raise RuntimeError(f"Image build failed: {e}")
        finally:
            # Clean up temporary Dockerfile
            if dockerfile_path.exists():
                dockerfile_path.unlink()
                self.logger.debug("Temporary Dockerfile cleaned up")