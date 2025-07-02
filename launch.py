import subprocess
import sys
import os

def main():
    # Number of parallel copies; default to 5 if not supplied
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    procs = []
    for i in range(n):
        # This will launch exactly the same script with the same env
        p = subprocess.Popen(
            [sys.executable, "smart-temp-TAKEOVER.py"],
            env=os.environ  # inherit your PGHOST, PGUSER, etc.
        )
        procs.append(p)
    
    # Wait for all of them to finish
    for p in procs:
        p.wait()
    
    print(f"✅ All {n} optimizer runs have completed.")

if __name__ == "__main__":
    main() 