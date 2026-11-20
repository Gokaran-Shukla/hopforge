"""20-question multiple-choice quizzes for every tutorial module.

>>> import hopforge as hf
>>> hf.Quiz.list()
>>> q = hf.Quiz("05")          # by number or name
>>> q.interactive()            # clickable quiz in Jupyter (needs ipywidgets)
>>> q.show()                   # plain-text questions
>>> q.check({1: "b", 2: "d"})  # grade your answers, with explanations
"""
from __future__ import annotations

import importlib

_GS_NOTICE = "© 2026 Gokaran Shukla · github.com/Gokaran-Shukla/hopforge · MIT License"  # gs-watermark


_MODULES = {
    "01": "q01_lattices", "02": "q02_bands", "03": "q03_geometry", "04": "q04_models",
    "05": "q05_berry_chern", "06": "q06_z2_wilson", "07": "q07_kubo",
    "08": "q08_scattering", "09": "q09_multiterminal", "10": "q10_superconductors_wavefunctions",
    "11": "q11_kpm_continuum", "12": "q12_parallel_params", "13": "q13_wannier_surface",
    "14": "q14_boltzmann_edelstein", "15": "q15_nonlinear_orbital_optical", "16": "q16_meanfield_floquet",
    "17": "q17_build_hamiltonian", "18": "q18_orbitronics_bilayers", "19": "q19_interacting_phases",
    "20": "q20_research_models",
    "21": "q21_macdonald_models",
    "22": "q22_dmrg", "23": "q23_negf", "24": "q24_gpu_registry",
}
LETTERS = "abcdefgh"


class Quiz:
    def __init__(self, module):
        key = str(module).zfill(2) if str(module).isdigit() else None
        if key is None:
            for k, v in _MODULES.items():
                if str(module).lower() in v:
                    key = k
                    break
        if key not in _MODULES:
            raise KeyError(f"unknown quiz {module!r}; available: {Quiz.list(quiet=True)}")
        mod = importlib.import_module(f"hopforge.quizzes.{_MODULES[key]}")
        self.key = key
        self.title = mod.TITLE
        self.questions = self._shuffled(key, mod.QUESTIONS)

    @staticmethod
    def _shuffled(key, questions):
        """Shuffle the options of every question reproducibly so the correct
        letter is not predictable."""
        import random
        out = []
        for n, (q, opts, ans, expl) in enumerate(questions, start=1):
            order = list(range(len(opts)))
            random.Random(f"hopforge-{key}-{n}").shuffle(order)
            new_opts = [opts[i] for i in order]
            new_ans = LETTERS[order.index(LETTERS.index(ans))]
            out.append((q, new_opts, new_ans, expl))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def list(quiet=False):
        out = []
        for k, v in _MODULES.items():
            mod = importlib.import_module(f"hopforge.quizzes.{v}")
            out.append(f"{k}: {mod.TITLE} ({len(mod.QUESTIONS)} questions)")
        if not quiet:
            print("\n".join(out))
        return out

    def __len__(self):
        return len(self.questions)

    def __repr__(self):
        return f"<Quiz {self.key}: {self.title}, {len(self)} questions>"

    def show(self, number=None):
        """Print all questions (or one, by 1-based number)."""
        qs = range(1, len(self) + 1) if number is None else [number]
        print(f"=== Quiz {self.key}: {self.title} ===\n")
        print(_GS_NOTICE + "\n")
        for n in qs:
            q, opts, _, _ = self.questions[n - 1]
            print(f"Q{n}. {q}")
            for i, o in enumerate(opts):
                print(f"   ({LETTERS[i]}) {o}")
            print()

    def check(self, answers, show_explanations=True):
        """Grade answers given as {question_number: letter} or a list of letters.
        Returns the score."""
        if isinstance(answers, (list, tuple, str)):
            answers = {n + 1: a for n, a in enumerate(answers)}
        score, lines = 0, []
        for n, (q, opts, ans, expl) in enumerate(self.questions, start=1):
            given = str(answers.get(n, "")).strip().lower()
            if not given:
                lines.append(f"Q{n}: (no answer)")
                continue
            if given == ans:
                score += 1
                lines.append(f"Q{n}: correct")
            else:
                msg = f"Q{n}: you chose ({given}), correct is ({ans}) {opts[LETTERS.index(ans)]}"
                if show_explanations:
                    msg += f"\n      -> {expl}"
                lines.append(msg)
        print("\n".join(lines))
        print(f"\nScore: {score} / {len(self)}" + ("  - excellent!" if score >= 18 else ""))
        return score

    def answer_key(self):
        """Return the answer key {number: letter} (for instructors)."""
        return {n: a for n, (_, _, a, _) in enumerate(self.questions, start=1)}

    def explain(self, number):
        q, opts, ans, expl = self.questions[number - 1]
        print(f"Q{number}. {q}\nAnswer: ({ans}) {opts[LETTERS.index(ans)]}\n{expl}")

    def interactive(self):
        """Clickable quiz for Jupyter (falls back to show() without ipywidgets)."""
        try:
            import ipywidgets as W
            from IPython.display import display
        except ImportError:
            print("ipywidgets not installed - showing the text version; grade with .check()")
            return self.show()
        boxes, feedback = [], []
        items = [W.HTML(f"<h3>Quiz {self.key}: {self.title}</h3>")]
        for n, (q, opts, ans, expl) in enumerate(self.questions, start=1):
            rb = W.RadioButtons(options=[f"({LETTERS[i]}) {o}" for i, o in enumerate(opts)],
                                value=None, layout=W.Layout(width="100%"))
            fb = W.HTML("")
            boxes.append(rb)
            feedback.append(fb)
            items += [W.HTML(f"<b>Q{n}.</b> {q}"), rb, fb]
        btn = W.Button(description="Grade my answers", button_style="primary")
        result = W.HTML("")

        def grade(_):
            score = 0
            for n, (rb, fb) in enumerate(zip(boxes, feedback)):
                _, opts, ans, expl = self.questions[n]
                if rb.value is None:
                    fb.value = "<span style='color:gray'>not answered</span>"
                    continue
                if rb.value.startswith(f"({ans})"):
                    score += 1
                    fb.value = "<span style='color:green'>&#10004; correct</span>"
                else:
                    fb.value = (f"<span style='color:#c00'>&#10008; correct answer ({ans}). "
                                f"{expl}</span>")
            result.value = f"<h3>Score: {score} / {len(self)}</h3>"
        btn.on_click(grade)
        display(W.VBox(items + [btn, result]))
