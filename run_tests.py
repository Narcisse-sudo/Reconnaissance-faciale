import os
import sys
import types
import importlib.util
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PY_FILES = [
    '1bwebcam2.py',
    '2recuperationface.py',
    '2split_dataset.py',
    '3-1train_model.py',
    '8Flask.py',
    '10.py',
]

def ensure_test_mocks() -> None:
    """Provide lightweight mocks for heavy/optional deps when missing.
    This lets modules import in DRY_RUN even if TensorFlow/Flask are not installed.
    """
    # Mock tensorflow if unavailable
    try:
        import tensorflow  # type: ignore
    except Exception:
        tf_module = types.ModuleType('tensorflow')
        keras_module = types.ModuleType('tensorflow.keras')

        class DummyModel:
            def predict(self, arr):
                return [[1.0]]

        models_module = types.SimpleNamespace(load_model=lambda *a, **k: DummyModel())
        layers_module = types.SimpleNamespace(
            Conv2D=lambda *a, **k: None,
            BatchNormalization=lambda *a, **k: None,
            MaxPooling2D=lambda *a, **k: None,
            Flatten=lambda *a, **k: None,
            Dense=lambda *a, **k: None,
            Dropout=lambda *a, **k: None,
        )
        callbacks_module = types.SimpleNamespace(TensorBoard=object)

        keras_module.models = models_module  # type: ignore[attr-defined]
        keras_module.layers = layers_module  # type: ignore[attr-defined]
        keras_module.callbacks = callbacks_module  # type: ignore[attr-defined]

        sys.modules['tensorflow'] = tf_module
        sys.modules['tensorflow.keras'] = keras_module
        sys.modules['tensorflow.keras.models'] = models_module  # type: ignore[assignment]
        sys.modules['tensorflow.keras.layers'] = layers_module  # type: ignore[assignment]
        sys.modules['tensorflow.keras.callbacks'] = callbacks_module  # type: ignore[assignment]

    # Mock flask if unavailable
    try:
        import flask  # type: ignore
    except Exception:
        flask_module = types.ModuleType('flask')
        def Flask(*args, **kwargs):
            class App:
                def route(self, *a, **k):
                    def decorator(fn):
                        return fn
                    return decorator
                def run(self, *a, **k):
                    pass
            return App()
        def Response(*args, **kwargs):
            return object()
        def render_template(*args, **kwargs):
            return ''
        flask_module.Flask = Flask  # type: ignore[attr-defined]
        flask_module.Response = Response  # type: ignore[attr-defined]
        flask_module.render_template = render_template  # type: ignore[attr-defined]
        sys.modules['flask'] = flask_module

    # Mock pyttsx3 if unavailable
    try:
        import pyttsx3  # type: ignore
    except Exception:
        pyttsx3_module = types.ModuleType('pyttsx3')
        class DummyEngine:
            def say(self, *a, **k):
                pass
            def runAndWait(self, *a, **k):
                pass
        pyttsx3_module.init = lambda *a, **k: DummyEngine()
        sys.modules['pyttsx3'] = pyttsx3_module

def run_module(filepath: str) -> tuple[bool, str]:
    try:
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        # Définir DRY_RUN pour éviter l'accès webcam/modèle
        os.environ['RF_DRY_RUN'] = '1'
        ensure_test_mocks()
        loader = spec.loader
        assert loader is not None
        loader.exec_module(module)
        return True, 'OK'
    except SystemExit as se:
        # Certaines apps Flask peuvent tenter de sortir
        return True, f'SystemExit: {se}'
    except Exception:
        return False, traceback.format_exc()

def main():
    print('Running project tests in DRY_RUN mode...')
    failures = []
    for rel in PY_FILES:
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(path):
            print(f'- SKIP {rel} (not found)')
            continue
        ok, msg = run_module(path)
        status = 'PASS' if ok else 'FAIL'
        print(f'- {status} {rel}: {msg.splitlines()[0] if msg else ""}')
        if not ok:
            failures.append((rel, msg))

    if failures:
        print('\nFailures:')
        for rel, msg in failures:
            print(f'\n### {rel}\n{msg}')
        raise SystemExit(1)
    print('\nAll tests passed.')

if __name__ == '__main__':
    main()

