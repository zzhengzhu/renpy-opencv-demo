init python:
    import os, json, subprocess, time, tempfile

    def run_fer_helper_once():
        import renpy

        # Ask Ren'Py for a *file* inside the bundle (reliable)
        plist_path = renpy.loader.transfn("helper/fer_helper.app/Contents/Info.plist")

        # Convert .../Contents/Info.plist -> .../fer_helper.app
        helper_app = plist_path.split("/Contents/")[0]

        # --- Concrete, predictable output path (inside the game folder) ---
        # We put results under game/helper/outputs/ so it's easy to find.
        base_game_dir = os.path.dirname(renpy.loader.transfn("script.rpy"))
        outputs_dir = os.path.join(base_game_dir, "helper", "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        out_path = os.path.join(outputs_dir, "fer_result.json")
        try:
            os.remove(out_path)
        except FileNotFoundError:
            pass

        subprocess.Popen(
            ["open", "-n", helper_app, "--args", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for _ in range(50):  # ~5 seconds
            if os.path.exists(out_path):
                break
            time.sleep(0.1)

        if not os.path.exists(out_path):
            return {"error": "no_output_file"}

        try:
            with open(out_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": "bad_output", "detail": str(e)}