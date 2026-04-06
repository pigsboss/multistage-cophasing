#!/usr/bin/env python3
"""
SPICE Kernel File Mirror Tool
Mirror NAIF generic_kernels directory to local storage
"""

import os
import sys
import time
import logging
import argparse
import concurrent.futures
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import requests
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SPICEMirror:
    """SPICE Kernel Mirror Tool"""
    
    def __init__(
        self,
        base_url: str = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/",
        local_dir: str = "./spice_mirror",
        max_workers: int = 5,
        skip_existing: bool = True,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize mirror tool
        
        Args:
            base_url: NAIF base URL
            local_dir: Local storage directory
            max_workers: Maximum concurrent downloads
            skip_existing: Skip existing files
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.base_url = base_url.rstrip('/') + '/'
        self.local_dir = Path(local_dir).resolve()
        self.max_workers = max_workers
        self.skip_existing = skip_existing
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Create session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MCPC-SPICE-Mirror/1.0 (for scientific research)'
        })
        
        # Create local directory
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_directory_listing(self, url: str) -> List[Dict]:
        """
        Get directory listing (parse HTML page)
        
        Args:
            url: Directory URL
            
        Returns:
            List of directory items, each as {'name': str, 'type': 'file'|'dir', 'size': int}
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Simple Apache directory listing parser
            # Note: This parser may need adjustment for different server configurations
            import re
            
            items = []
            
            # Match directory items (Apache directory listing format)
            # Pattern matches: <a href="de440.bsp">de440.bsp</a> or <a href="spk/">spk/</a>
            pattern = r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>\s+(\d{2,4}-\w{3}-\d{4}\s+\d{2}:\d{2})\s+([\d\.\-]+[KMG]?)'
            
            for match in re.finditer(pattern, response.text):
                name = match.group(1)
                display_name = match.group(2)
                date_str = match.group(3)
                size_str = match.group(4)
                
                # Skip parent directory links
                if name in ['../', '..', '.']:
                    continue
                
                # Determine if file or directory
                is_dir = name.endswith('/')
                name = name.rstrip('/')
                
                # Parse file size
                size = 0
                if size_str != '-':
                    # Handle 1.2K, 1.2M, 1.2G formats
                    size_lower = size_str.lower()
                    if 'k' in size_lower:
                        multiplier = 1024
                        num = float(size_lower.replace('k', ''))
                    elif 'm' in size_lower:
                        multiplier = 1024 * 1024
                        num = float(size_lower.replace('m', ''))
                    elif 'g' in size_lower:
                        multiplier = 1024 * 1024 * 1024
                        num = float(size_lower.replace('g', ''))
                    else:
                        multiplier = 1
                        num = float(size_lower)
                    size = int(num * multiplier)
                
                items.append({
                    'name': name,
                    'display_name': display_name,
                    'type': 'dir' if is_dir else 'file',
                    'size': size,
                    'date': date_str
                })
            
            return items
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get directory listing {url}: {e}")
            return []
    
    def _download_file(self, url: str, local_path: Path) -> bool:
        """
        Download single file
        
        Args:
            url: File URL
            local_path: Local save path
            
        Returns:
            Success status
        """
        # Check if file exists and skip
        if self.skip_existing and local_path.exists():
            local_size = local_path.stat().st_size
            try:
                # Get remote file size
                head_resp = self.session.head(url, timeout=self.timeout, allow_redirects=True)
                remote_size = int(head_resp.headers.get('content-length', 0))
                
                if local_size == remote_size:
                    logger.debug(f"Skipping existing file: {local_path}")
                    return True
            except:
                pass  # If HEAD request fails, continue with download
        
        # Create directory
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                # Stream download
                response = self.session.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                # Write file
                with open(local_path, 'wb') as f:
                    if total_size == 0:
                        f.write(response.content)
                    else:
                        # Show progress bar
                        with tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=local_path.name,
                            leave=False
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                
                logger.info(f"Download completed: {local_path}")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download failed (attempt {attempt+1}/{self.max_retries}): {url} - {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
                # Delete incomplete file
                if local_path.exists():
                    local_path.unlink()
        
        logger.error(f"Download failed, exceeded maximum retries: {url}")
        return False
    
    def _mirror_directory(self, remote_path: str, local_path: Path, depth: int = 0) -> Dict:
        """
        Recursively mirror directory
        
        Args:
            remote_path: Remote directory path (relative to base_url)
            local_path: Local directory path
            depth: Recursion depth
            
        Returns:
            Statistics
        """
        stats = {'directories': 0, 'files': 0, 'size': 0, 'errors': 0}
        
        # Ensure local directory exists
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Get directory listing
        dir_url = urljoin(self.base_url, remote_path)
        logger.info(f"Scanning directory: {remote_path}")
        
        items = self._get_directory_listing(dir_url)
        if not items:
            logger.warning(f"Directory empty or inaccessible: {remote_path}")
            return stats
        
        # Process files first, then directories
        files_to_download = []
        dirs_to_process = []
        
        for item in items:
            item_url = urljoin(dir_url, item['name'])
            item_local_path = local_path / item['name']
            
            if item['type'] == 'file':
                files_to_download.append((item_url, item_local_path, item['size']))
            elif item['type'] == 'dir':
                dirs_to_process.append((item['name'], item_local_path))
        
        # Concurrent file downloads
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_file = {
                executor.submit(self._download_file, url, path): (url, path, size)
                for url, path, size in files_to_download
            }
            
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_file):
                url, path, size = future_to_file[future]
                try:
                    success = future.result()
                    if success:
                        stats['files'] += 1
                        stats['size'] += size
                    else:
                        stats['errors'] += 1
                except Exception as e:
                    logger.error(f"Download exception: {url} - {e}")
                    stats['errors'] += 1
        
        # Recursively process subdirectories
        for dir_name, dir_local_path in dirs_to_process:
            # Skip certain directories (optional)
            if depth > 10:  # Limit recursion depth
                logger.warning(f"Reached recursion depth limit, skipping: {dir_name}")
                continue
                
            sub_stats = self._mirror_directory(
                f"{remote_path.rstrip('/')}/{dir_name}",
                dir_local_path,
                depth + 1
            )
            
            # Merge statistics
            for key in stats:
                stats[key] += sub_stats[key]
            stats['directories'] += 1
        
        return stats
    
    def mirror(self, subpath: str = "") -> Dict:
        """
        Main mirror function
        
        Args:
            subpath: Subpath to mirror (empty for all)
            
        Returns:
            Statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting mirror: {self.base_url}")
        logger.info(f"Local directory: {self.local_dir}")
        logger.info(f"Concurrent workers: {self.max_workers}")
        logger.info(f"Skip existing: {self.skip_existing}")
        
        # Execute mirror
        stats = self._mirror_directory(
            subpath,
            self.local_dir / subpath if subpath else self.local_dir
        )
        
        # Statistics
        elapsed_time = time.time() - start_time
        stats['elapsed_time'] = elapsed_time
        
        logger.info(f"Mirror completed!")
        logger.info(f"Directories: {stats['directories']}")
        logger.info(f"Files: {stats['files']}")
        logger.info(f"Total size: {self._format_size(stats['size'])}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info(f"Time elapsed: {self._format_time(elapsed_time)}")
        
        # Generate index file
        self._generate_index()
        
        return stats
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def _format_time(self, seconds: float) -> str:
        """Format time duration"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        else:
            return f"{seconds/3600:.1f} hours"
    
    def _generate_index(self):
        """Generate index file"""
        index_file = self.local_dir / "MIRROR_INFO.txt"
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(f"# SPICE Kernel Mirror Information\n")
            f.write(f"Mirror time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source URL: {self.base_url}\n")
            f.write(f"Local path: {self.local_dir}\n")
            f.write(f"\n")
            f.write(f"# Usage Instructions\n")
            f.write(f"1. Add this directory to SPICE kernel path\n")
            f.write(f"2. Use spiceypy to load kernels:\n")
            f.write(f"   import spiceypy as sp\n")
            f.write(f"   sp.furnsh('path/to/kernel.tf')\n")
            f.write(f"\n")
            f.write(f"# Main Kernel Files\n")
            f.write(f"- lsk/naif0012.tls (Leap Second Kernel)\n")
            f.write(f"- fk/frames/frames.tf (Frame Definitions)\n")
            f.write(f"- fk/satellites/moon_080317.tf (Lunar coordinate system)\n")
            f.write(f"- spk/planets/de440.bsp (Planetary ephemeris, ~2GB)\n")
            f.write(f"- pck/pck00010.tpc (Planetary constants)\n")
            f.write(f"\n")
            f.write(f"# Notes\n")
            f.write(f"- Large files like de440.bsp require significant download time\n")
            f.write(f"- Run this script periodically to update kernels\n")
            f.write(f"- Delete unnecessary files to save space\n")

def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description='Mirror NAIF SPICE kernel files to local storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mirror entire generic_kernels directory (~100+ GB)
  python spice_mirror.py --all
  
  # Mirror specific subdirectory only
  python spice_mirror.py --subdir lsk
  python spice_mirror.py --subdir spk/planets
  
  # Specify local directory and concurrency
  python spice_mirror.py --all --local ./spice_data --workers 10
  
  # Do not skip existing files (force update)
  python spice_mirror.py --all --no-skip
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Mirror entire generic_kernels directory'
    )
    
    parser.add_argument(
        '--subdir',
        type=str,
        default='',
        help='Mirror only specified subdirectory (e.g., "lsk", "spk/planets")'
    )
    
    parser.add_argument(
        '--local',
        type=str,
        default='./spice_mirror',
        help='Local storage directory (default: ./spice_mirror)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='Concurrent download count (default: 5)'
    )
    
    parser.add_argument(
        '--no-skip',
        dest='skip_existing',
        action='store_false',
        help='Do not skip existing files (force re-download)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Maximum retry attempts (default: 3)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Check arguments
    if not args.all and not args.subdir:
        logger.error("Please specify either --all or --subdir argument")
        parser.print_help()
        sys.exit(1)
    
    # Path to mirror
    mirror_path = args.subdir if args.subdir else ""
    
    # Create mirror instance
    mirror = SPICEMirror(
        local_dir=args.local,
        max_workers=args.workers,
        skip_existing=args.skip_existing,
        timeout=args.timeout,
        max_retries=args.retries
    )
    
    try:
        # Execute mirror
        stats = mirror.mirror(mirror_path)
        
        # Show warnings
        if stats['errors'] > 0:
            logger.warning(f"Mirror completed with {stats['errors']} errors")
        
    except KeyboardInterrupt:
        logger.info("Mirror interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Mirror failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
