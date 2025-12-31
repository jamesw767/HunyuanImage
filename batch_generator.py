#!/usr/bin/env python3
"""
Batch Generator for HunyuanImage-3.0

Processes multiple prompts with queue management, progress tracking,
resume capability, and organized output.
"""

import os
import sys
import csv
import json
import time
import sqlite3
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Generator
from dataclasses import dataclass, asdict
import logging

# Add HunyuanImage to path
sys.path.insert(0, str(Path(__file__).parent / "HunyuanImage-3.0"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path("/media/james/DataDrive/jamesw767/Hun3d")
OUTPUT_DIR = BASE_DIR / "outputs"
BATCHES_DIR = OUTPUT_DIR / "batches"
QUEUE_DB = BASE_DIR / "batch_queue.db"

# Style presets (from hunyuan_ui.py)
STYLE_PRESETS = {
    "none": "",
    "photorealistic": ", photorealistic, hyperrealistic, 8k, highly detailed, professional photography",
    "cinematic": ", cinematic lighting, dramatic atmosphere, movie still, 35mm film grain",
    "anime": ", anime style, vibrant colors, detailed linework, studio quality anime",
    "digital_art": ", digital art, concept art, artstation trending, highly detailed illustration",
    "oil_painting": ", oil painting style, classical art, rich textures, museum quality",
    "fantasy": ", fantasy art, epic fantasy, magical atmosphere, detailed fantasy illustration",
    "3d_render": ", 3D render, octane render, unreal engine 5, high quality CGI",
}


@dataclass
class BatchJob:
    """A batch processing job"""
    batch_id: str
    name: str
    created_at: str
    status: str  # pending, running, paused, completed, failed
    total_prompts: int
    completed_prompts: int
    failed_prompts: int
    output_dir: str


@dataclass
class PromptItem:
    """A single prompt in the batch queue"""
    id: int
    batch_id: str
    prompt: str
    style: str
    seed: Optional[int]
    aspect_ratio: str
    steps: int
    status: str  # pending, processing, completed, failed
    output_path: Optional[str]
    error: Optional[str]
    generation_time: Optional[float]
    created_at: str
    completed_at: Optional[str]


class BatchQueue:
    """SQLite-backed batch processing queue"""

    def __init__(self, db_path: Path = QUEUE_DB):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT,
                status TEXT DEFAULT 'pending',
                total_prompts INTEGER DEFAULT 0,
                completed_prompts INTEGER DEFAULT 0,
                failed_prompts INTEGER DEFAULT 0,
                output_dir TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT,
                prompt TEXT,
                style TEXT DEFAULT 'none',
                seed INTEGER,
                aspect_ratio TEXT DEFAULT '1024x1024',
                steps INTEGER DEFAULT 20,
                status TEXT DEFAULT 'pending',
                output_path TEXT,
                error TEXT,
                generation_time REAL,
                created_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_batch_id ON prompts (batch_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON prompts (status)')

        conn.commit()
        conn.close()

    def create_batch(self, name: str, prompts: List[Dict]) -> str:
        """Create a new batch job with prompts"""
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        output_dir = str(BATCHES_DIR / batch_id)
        os.makedirs(output_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create batch
        cursor.execute('''
            INSERT INTO batches (batch_id, name, created_at, status, total_prompts, output_dir)
            VALUES (?, ?, ?, 'pending', ?, ?)
        ''', (batch_id, name, datetime.now().isoformat(), len(prompts), output_dir))

        # Add prompts
        for p in prompts:
            cursor.execute('''
                INSERT INTO prompts (batch_id, prompt, style, seed, aspect_ratio, steps, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                batch_id,
                p.get('prompt', ''),
                p.get('style', 'none'),
                p.get('seed'),
                p.get('aspect_ratio', '1024x1024'),
                p.get('steps', 20),
                datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()

        # Save manifest
        manifest = {
            "batch_id": batch_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "total_prompts": len(prompts),
            "prompts": prompts
        }
        with open(Path(output_dir) / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created batch {batch_id} with {len(prompts)} prompts")
        return batch_id

    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Get batch job info"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM batches WHERE batch_id = ?', (batch_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return BatchJob(
                batch_id=row[0], name=row[1], created_at=row[2], status=row[3],
                total_prompts=row[4], completed_prompts=row[5], failed_prompts=row[6],
                output_dir=row[7]
            )
        return None

    def list_batches(self, status: Optional[str] = None) -> List[BatchJob]:
        """List all batches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            cursor.execute('SELECT * FROM batches WHERE status = ? ORDER BY created_at DESC', (status,))
        else:
            cursor.execute('SELECT * FROM batches ORDER BY created_at DESC')

        batches = []
        for row in cursor.fetchall():
            batches.append(BatchJob(
                batch_id=row[0], name=row[1], created_at=row[2], status=row[3],
                total_prompts=row[4], completed_prompts=row[5], failed_prompts=row[6],
                output_dir=row[7]
            ))

        conn.close()
        return batches

    def get_pending_prompt(self, batch_id: str) -> Optional[PromptItem]:
        """Get next pending prompt from batch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM prompts
            WHERE batch_id = ? AND status = 'pending'
            ORDER BY id LIMIT 1
        ''', (batch_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return PromptItem(
                id=row[0], batch_id=row[1], prompt=row[2], style=row[3],
                seed=row[4], aspect_ratio=row[5], steps=row[6], status=row[7],
                output_path=row[8], error=row[9], generation_time=row[10],
                created_at=row[11], completed_at=row[12]
            )
        return None

    def update_prompt_status(
        self,
        prompt_id: int,
        status: str,
        output_path: Optional[str] = None,
        error: Optional[str] = None,
        generation_time: Optional[float] = None
    ):
        """Update prompt status after processing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        completed_at = datetime.now().isoformat() if status in ('completed', 'failed') else None

        cursor.execute('''
            UPDATE prompts
            SET status = ?, output_path = ?, error = ?, generation_time = ?, completed_at = ?
            WHERE id = ?
        ''', (status, output_path, error, generation_time, completed_at, prompt_id))

        # Update batch counters
        cursor.execute('SELECT batch_id FROM prompts WHERE id = ?', (prompt_id,))
        batch_id = cursor.fetchone()[0]

        if status == 'completed':
            cursor.execute('''
                UPDATE batches SET completed_prompts = completed_prompts + 1
                WHERE batch_id = ?
            ''', (batch_id,))
        elif status == 'failed':
            cursor.execute('''
                UPDATE batches SET failed_prompts = failed_prompts + 1
                WHERE batch_id = ?
            ''', (batch_id,))

        conn.commit()
        conn.close()

    def update_batch_status(self, batch_id: str, status: str):
        """Update batch status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE batches SET status = ? WHERE batch_id = ?', (status, batch_id))
        conn.commit()
        conn.close()

    def get_batch_progress(self, batch_id: str) -> Tuple[int, int, int]:
        """Get batch progress (completed, failed, total)"""
        batch = self.get_batch(batch_id)
        if batch:
            return batch.completed_prompts, batch.failed_prompts, batch.total_prompts
        return 0, 0, 0

    def reset_failed_prompts(self, batch_id: str) -> int:
        """Reset failed prompts to pending for retry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE prompts SET status = 'pending', error = NULL
            WHERE batch_id = ? AND status = 'failed'
        ''', (batch_id,))

        count = cursor.rowcount

        cursor.execute('''
            UPDATE batches SET failed_prompts = 0
            WHERE batch_id = ?
        ''', (batch_id,))

        conn.commit()
        conn.close()

        logger.info(f"Reset {count} failed prompts to pending")
        return count


def load_prompts_from_file(filepath: str) -> Tuple[str, List[Dict]]:
    """Load prompts from CSV, JSON, or TXT file"""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    prompts = []
    name = path.stem

    if path.suffix.lower() == '.csv':
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompts.append({
                    'prompt': row.get('prompt', ''),
                    'style': row.get('style', 'none'),
                    'seed': int(row['seed']) if row.get('seed') else None,
                    'aspect_ratio': row.get('aspect_ratio', '1024x1024'),
                    'steps': int(row.get('steps', 20))
                })

    elif path.suffix.lower() == '.json':
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict):
            name = data.get('batch_name', name)
            prompts = data.get('prompts', [])
        elif isinstance(data, list):
            prompts = data

        # Normalize prompt format
        for i, p in enumerate(prompts):
            if isinstance(p, str):
                prompts[i] = {'prompt': p}
            else:
                prompts[i] = {
                    'prompt': p.get('prompt', ''),
                    'style': p.get('style', 'none'),
                    'seed': p.get('seed'),
                    'aspect_ratio': p.get('aspect_ratio', '1024x1024'),
                    'steps': p.get('steps', 20)
                }

    elif path.suffix.lower() == '.txt':
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    prompts.append({'prompt': line})

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return name, prompts


def apply_style_to_prompt(prompt: str, style: str) -> str:
    """Apply style preset to prompt"""
    style_suffix = STYLE_PRESETS.get(style.lower(), "")
    if style_suffix and not prompt.endswith(style_suffix):
        return prompt.rstrip('.') + style_suffix
    return prompt


class BatchProcessor:
    """Process batch jobs with HunyuanImage model"""

    def __init__(self, enhance_prompts: bool = False, ollama_model: str = "qwen2.5:7b-instruct"):
        self.model = None
        self.model_loaded = False
        self.queue = BatchQueue()
        self.enhance_prompts = enhance_prompts
        self.ollama_model = ollama_model
        self.enhancer = None

    def load_model(self):
        """Load the HunyuanImage model"""
        if self.model_loaded:
            return

        import torch
        from transformers import AutoModelForCausalLM
        from sdnq import SDNQConfig

        model_id = str(BASE_DIR / "HunyuanImage3-SDNQ")

        logger.info("Loading quantized HunyuanImage-3.0 model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            moe_impl="eager",
        )
        self.model.load_tokenizer(model_id)
        self.model_loaded = True

        logger.info("Model loaded successfully!")

        # Initialize enhancer if needed
        if self.enhance_prompts:
            from ollama_prompts import PromptEnhancer
            self.enhancer = PromptEnhancer(model=self.ollama_model)
            logger.info(f"Prompt enhancement enabled with {self.ollama_model}")

    def generate_single(
        self,
        prompt: str,
        seed: Optional[int] = None,
        aspect_ratio: str = "1024x1024",
        steps: int = 20,
        style: str = "none"
    ) -> Tuple[any, float]:
        """Generate a single image"""
        if not self.model_loaded:
            self.load_model()

        # Apply style
        styled_prompt = apply_style_to_prompt(prompt, style)

        # Enhance with Ollama if enabled
        if self.enhancer:
            try:
                styled_prompt = self.enhancer.enhance(styled_prompt, temperature=0.7)
            except Exception as e:
                logger.warning(f"Enhancement failed, using original: {e}")

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        start_time = time.time()

        image = self.model.generate_image(
            prompt=styled_prompt,
            seed=seed,
            image_size=aspect_ratio,
            diff_infer_steps=steps,
            stream=True,
        )

        generation_time = time.time() - start_time

        return image, generation_time, seed

    def process_batch(
        self,
        batch_id: str,
        retry_failed: bool = False,
        max_retries: int = 3
    ) -> Generator[Tuple[int, int, str], None, None]:
        """
        Process a batch job.

        Yields: (current, total, status_message)
        """
        batch = self.queue.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch not found: {batch_id}")

        output_dir = Path(batch.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if retry_failed:
            self.queue.reset_failed_prompts(batch_id)

        self.queue.update_batch_status(batch_id, 'running')

        # Load model
        if not self.model_loaded:
            yield (0, batch.total_prompts, "Loading model...")
            self.load_model()

        completed, failed, total = self.queue.get_batch_progress(batch_id)
        current = completed + failed

        while True:
            item = self.queue.get_pending_prompt(batch_id)
            if not item:
                break

            current += 1
            yield (current, total, f"Processing: {item.prompt[:50]}...")

            self.queue.update_prompt_status(item.id, 'processing')

            retry_count = 0
            success = False

            while retry_count < max_retries and not success:
                try:
                    image, gen_time, seed = self.generate_single(
                        prompt=item.prompt,
                        seed=item.seed,
                        aspect_ratio=item.aspect_ratio,
                        steps=item.steps,
                        style=item.style
                    )

                    # Save image
                    safe_prompt = "".join(c for c in item.prompt[:30] if c.isalnum() or c in " -_").strip()
                    safe_prompt = safe_prompt.replace(" ", "_")
                    filename = f"{current:04d}_{safe_prompt}_s{seed}.png"
                    filepath = output_dir / filename

                    image.save(filepath)

                    self.queue.update_prompt_status(
                        item.id, 'completed',
                        output_path=str(filepath),
                        generation_time=gen_time
                    )

                    logger.info(f"[{current}/{total}] Generated: {filename} ({gen_time:.1f}s)")
                    success = True

                except Exception as e:
                    retry_count += 1
                    logger.error(f"Error (attempt {retry_count}/{max_retries}): {e}")

                    if retry_count >= max_retries:
                        self.queue.update_prompt_status(
                            item.id, 'failed',
                            error=str(e)
                        )
                        logger.error(f"[{current}/{total}] Failed: {item.prompt[:50]}...")

        # Update batch status
        completed, failed, total = self.queue.get_batch_progress(batch_id)
        if failed == 0:
            self.queue.update_batch_status(batch_id, 'completed')
            yield (total, total, f"Batch completed! {completed} images generated.")
        else:
            self.queue.update_batch_status(batch_id, 'completed')
            yield (total, total, f"Batch completed with {failed} failures. {completed} images generated.")

        # Save summary
        summary = {
            "batch_id": batch_id,
            "completed_at": datetime.now().isoformat(),
            "total": total,
            "completed": completed,
            "failed": failed,
        }
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """CLI interface for batch processing"""
    parser = argparse.ArgumentParser(description="Batch Generator for HunyuanImage-3.0")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create batch
    create_parser = subparsers.add_parser('create', help='Create a new batch from file')
    create_parser.add_argument('file', help='Input file (CSV, JSON, or TXT)')
    create_parser.add_argument('--name', '-n', help='Batch name')
    create_parser.add_argument('--enhance', action='store_true', help='Enhance prompts with Ollama')
    create_parser.add_argument('--model', '-m', default='qwen2.5:7b-instruct', help='Ollama model for enhancement')

    # Run batch
    run_parser = subparsers.add_parser('run', help='Run a batch job')
    run_parser.add_argument('batch_id', nargs='?', help='Batch ID to run (latest if not specified)')
    run_parser.add_argument('--retry', action='store_true', help='Retry failed prompts')
    run_parser.add_argument('--enhance', action='store_true', help='Enhance prompts with Ollama')
    run_parser.add_argument('--model', '-m', default='qwen2.5:7b-instruct', help='Ollama model for enhancement')

    # List batches
    list_parser = subparsers.add_parser('list', help='List batch jobs')
    list_parser.add_argument('--status', '-s', help='Filter by status')

    # Show batch
    show_parser = subparsers.add_parser('show', help='Show batch details')
    show_parser.add_argument('batch_id', help='Batch ID')

    # Resume batch
    resume_parser = subparsers.add_parser('resume', help='Resume a paused/interrupted batch')
    resume_parser.add_argument('batch_id', help='Batch ID to resume')
    resume_parser.add_argument('--enhance', action='store_true', help='Enhance prompts with Ollama')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    queue = BatchQueue()

    if args.command == 'create':
        name, prompts = load_prompts_from_file(args.file)
        if args.name:
            name = args.name

        if args.enhance:
            from ollama_prompts import PromptEnhancer
            enhancer = PromptEnhancer(model=args.model)
            print(f"Enhancing {len(prompts)} prompts with {args.model}...")

            for i, p in enumerate(prompts):
                try:
                    enhanced = enhancer.enhance(p['prompt'])
                    print(f"[{i+1}/{len(prompts)}] Enhanced: {p['prompt'][:40]}...")
                    p['prompt'] = enhanced
                except Exception as e:
                    print(f"[{i+1}/{len(prompts)}] Enhancement failed: {e}")

        batch_id = queue.create_batch(name, prompts)
        print(f"\nBatch created: {batch_id}")
        print(f"Total prompts: {len(prompts)}")
        print(f"\nRun with: python batch_generator.py run {batch_id}")

    elif args.command == 'run':
        batch_id = args.batch_id
        if not batch_id:
            batches = queue.list_batches('pending')
            if not batches:
                print("No pending batches found.")
                return
            batch_id = batches[0].batch_id
            print(f"Running latest batch: {batch_id}")

        processor = BatchProcessor(
            enhance_prompts=args.enhance,
            ollama_model=args.model
        )

        for current, total, status in processor.process_batch(batch_id, retry_failed=args.retry):
            print(f"[{current}/{total}] {status}")

    elif args.command == 'list':
        batches = queue.list_batches(args.status)
        if not batches:
            print("No batches found.")
            return

        print(f"\n{'Batch ID':<35} {'Name':<20} {'Status':<12} {'Progress'}")
        print("-" * 85)
        for b in batches:
            progress = f"{b.completed_prompts}/{b.total_prompts}"
            if b.failed_prompts > 0:
                progress += f" ({b.failed_prompts} failed)"
            print(f"{b.batch_id:<35} {b.name[:20]:<20} {b.status:<12} {progress}")

    elif args.command == 'show':
        batch = queue.get_batch(args.batch_id)
        if not batch:
            print(f"Batch not found: {args.batch_id}")
            return

        print(f"\nBatch: {batch.batch_id}")
        print(f"Name: {batch.name}")
        print(f"Status: {batch.status}")
        print(f"Created: {batch.created_at}")
        print(f"Progress: {batch.completed_prompts}/{batch.total_prompts}")
        print(f"Failed: {batch.failed_prompts}")
        print(f"Output: {batch.output_dir}")

    elif args.command == 'resume':
        processor = BatchProcessor(enhance_prompts=args.enhance)

        for current, total, status in processor.process_batch(args.batch_id):
            print(f"[{current}/{total}] {status}")


if __name__ == "__main__":
    main()
