import asyncio
import navel
import requests
import random
import select
import sys
import termios
import tty
from threading import Thread

class RobotManager:
    """Singleton manager for navel.Robot to share between modules."""
    def __init__(self):
        self.robot: navel.Robot | None = None
        self._lock = asyncio.Lock()  # prevent concurrent Robot commands
        self._running = True

    async def initialize(self):
        if self.robot:
            return

        print("[RobotManager] Initializing robot...")
        self.robot = await navel.Robot().__aenter__()

        # Configure FaceController params
        self.robot.config_set("cns_fer_cont_t1", 0, navel.DataType.U32)
        self.robot.config_set("cns_fer_cont_t2", 0, navel.DataType.U32)
        self.robot.config_set("cns_fer_peak_t1", 0, navel.DataType.U32)
        self.robot.config_set("cns_fer_peak_t2", 0, navel.DataType.U32)
        self.robot.config_set("cns_fer_overlay_inc", 0, navel.DataType.F32)
        self.robot.config_set("cns_fer_overlay_max", 0, navel.DataType.F32)
        self.robot.config_set("cns_fer_overlay_peak_min", 0, navel.DataType.F32)
        self.robot.config_set("cns_fer_peak_max", 0, navel.DataType.F32)
        print("[RobotManager] Robot initialized")

    async def shutdown(self):
        print("[RobotManager] Shutting down robot...")
        self._running = False
        if self.robot:
            await self.robot.__aexit__(None, None, None)
            self.robot = None
        print("[RobotManager] Shutdown complete")

    async def robot_say_emotion(self, text, emotion):
        if emotion and emotion.lower()=="happy":
            self.robot.head_facial_expression(0,0.7,0,0,0,0.3)
        elif emotion and emotion.lower()=="neutral":
            self.robot.head_facial_expression(1,0,0,0,0,0)
        elif emotion and emotion.lower() in ["surprised", "fear","surprise"]:
            self.robot.head_facial_expression(0,0,0,1,0,0)
        elif emotion and emotion.lower()=="sad":
            self.robot.head_facial_expression(0,0,1,0,0,0)
        elif emotion and emotion.lower() in ["angry"]:
            print("disgusted face")
            self.robot.head_facial_expression(0,0,0,0,1,0)
        elif emotion and emotion.lower() == "disgust":
            print("Disgust face")
            self.robot.head_facial_expression(0,0.05, 0.12, 0.83, 0, 0)
        else:
            print("default neutral face")
            self.robot.head_facial_expression(1,0,0,0,0,0)
        # robot.say() is async and waits for speech to complete
        await self.robot.say("<lang,it1>"+text)

    async def process_move_command(self, key: str):
        if key in ('w', 'up'):
            print("[MOVE] forward")
            await self.robot.move_base(0.25, 0.5)
        elif key in ('s', 'down'):
            print("[MOVE] backward")
            await self.robot.move_base(-0.25, 0.5)
        elif key in ('a', 'left'):
            print("[MOVE] turn left")
            await self.robot.rotate_base(20,40)
        elif key in ('d', 'right'):
            print("[MOVE] turn right")
            await self.robot.rotate_base(-20,40)

    async def terminal_controller(self):
        if not sys.stdin.isatty():
            print("[TERMINAL] stdin non è un terminale interattivo, controller non disponibile")
            return

        loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._terminal_input_loop, loop)

    def _terminal_input_loop(self, loop):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        print("[TERMINAL] Controllo attivo. Usa WASD o frecce, q per uscire.")
        try:
            while self._running:
                r, _, _ = select.select([sys.stdin], [], [], 0.01)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    break

                key = None
                if ch == '\x1b':
                    seq = sys.stdin.read(2)
                    if seq == '[A':
                        key = 'up'
                    elif seq == '[B':
                        key = 'down'
                    elif seq == '[C':
                        key = 'right'
                    elif seq == '[D':
                        key = 'left'
                else:
                    if ch.lower() in ('w', 'a', 's', 'd', 'q'):
                        key = ch.lower()

                if key == 'q':
                    print("[TERMINAL] Comando di uscita ricevuto")
                    self._running = False
                    break
                if key:
                    loop.call_soon_threadsafe(asyncio.create_task, self.process_move_command(key))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ---------------------------
# Example main
# ---------------------------
async def main():
    robot_manager = RobotManager()
    await robot_manager.initialize()
    emotions = ["happy", "neutral", "sad", "angry", "fear", "disgust", "surprise"]
    await robot_manager.terminal_controller()
    await robot_manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())