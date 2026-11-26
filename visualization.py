import numpy as np
from pathlib import Path


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

FIGURES_DIR.mkdir(exist_ok=True)


def save_svg(filename, content, width=900, height=500):

    svg = f'''
    <svg xmlns="http://www.w3.org/2000/svg"
         width="{width}" height="{height}"
         viewBox="0 0 {width} {height}">

        <rect width="100%" height="100%" fill="white"/>

        <g font-family="Arial, sans-serif">
            {content}
        </g>

    </svg>
    '''

    path = FIGURES_DIR / filename

    path.write_text(svg, encoding="utf-8")

    print(f"Created: {path}")


def plot_training_loss():

    data = np.loadtxt(
        RESULTS_DIR / "training_loss.csv",
        delimiter=",",
        skiprows=1
    )

    epochs = data[:, 0]
    losses = data[:, 1]

    left, right = 90, 850
    top, bottom = 70, 420

    max_loss = np.ceil(losses.max() * 20) / 20

    x = left + epochs / epochs.max() * (right - left)

    y = bottom - losses / max_loss * (bottom - top)

    points = " ".join(
        f"{xi:.2f},{yi:.2f}"
        for xi, yi in zip(x, y)
    )

    content = '''
    <text x="450" y="35" text-anchor="middle"
          font-size="23" font-weight="bold">
        Training Loss - Mean Squared Error
    </text>
    '''

    # Horizontal grid and MSE labels
    for value in np.linspace(0, max_loss, 7):

        py = bottom - value / max_loss * (bottom - top)

        content += f'''
        <line x1="{left}" y1="{py}"
              x2="{right}" y2="{py}"
              stroke="#e2e8f0"/>

        <text x="78" y="{py + 4}"
              text-anchor="end" font-size="12">
            {value:.2f}
        </text>
        '''

    # Epoch labels
    for epoch in np.linspace(0, epochs.max(), 6):

        px = left + epoch / epochs.max() * (right - left)

        content += f'''
        <text x="{px}" y="445"
              text-anchor="middle" font-size="12">
            {int(epoch)}
        </text>
        '''

    # Axes and loss curve
    content += f'''
    <line x1="{left}" y1="{top}"
          x2="{left}" y2="{bottom}" stroke="#334155"/>

    <line x1="{left}" y1="{bottom}"
          x2="{right}" y2="{bottom}" stroke="#334155"/>

    <polyline points="{points}" fill="none"
              stroke="#2563eb" stroke-width="3"/>

    <text x="470" y="480" text-anchor="middle"
          font-size="15">Epoch</text>

    <text x="25" y="245" text-anchor="middle"
          transform="rotate(-90 25 245)"
          font-size="15">MSE</text>

    <text x="830" y="95" text-anchor="end"
          font-size="14">
        Initial MSE: {losses[0]:.6f}
    </text>

    <text x="830" y="118" text-anchor="end"
          font-size="14">
        Final MSE: {losses[-1]:.6f}
    </text>
    '''

    save_svg("training_loss.svg", content)


def plot_confusion_matrix():

    matrix = np.loadtxt(
        RESULTS_DIR / "confusion_matrix.csv",
        delimiter=",",
        skiprows=1,
        dtype=int
    )

    content = '''
    <text x="350" y="35" text-anchor="middle"
          font-size="24" font-weight="bold">
        Confusion Matrix
    </text>

    <text x="400" y="75" text-anchor="middle"
          font-size="16">Predicted Class</text>

    <text x="300" y="105" text-anchor="middle">
        Class 0
    </text>

    <text x="500" y="105" text-anchor="middle">
        Class 1
    </text>

    <text x="45" y="320" text-anchor="middle"
          transform="rotate(-90 45 320)"
          font-size="16">Actual Class</text>

    <text x="145" y="225" text-anchor="middle">
        Class 0
    </text>

    <text x="145" y="425" text-anchor="middle">
        Class 1
    </text>
    '''

    labels = [
        ["TN", "FP"],
        ["FN", "TP"]
    ]

    for row in range(2):

        for col in range(2):

            value = matrix[row, col]

            x = 200 + col * 200
            y = 120 + row * 200

            opacity = 0.08 + 0.92 * value / max(
                matrix.max(), 1
            )

            text_color = (
                "white" if opacity > 0.5 else "#1e293b"
            )

            content += f'''
            <rect x="{x}" y="{y}"
                  width="195" height="195" rx="5"
                  fill="#2563eb"
                  fill-opacity="{opacity}"/>

            <text x="{x + 97.5}" y="{y + 95}"
                  text-anchor="middle"
                  font-size="40" font-weight="bold"
                  fill="{text_color}">
                {value}
            </text>

            <text x="{x + 97.5}" y="{y + 130}"
                  text-anchor="middle"
                  font-size="18" fill="{text_color}">
                {labels[row][col]}
            </text>
            '''

    save_svg(
        "confusion_matrix.svg",
        content,
        width=700,
        height=560
    )


def plot_accuracy_comparison():

    data = np.loadtxt(
        RESULTS_DIR / "metrics.csv",
        delimiter=",",
        skiprows=1,
        dtype=str
    )

    metrics = {
        row[0]: float(row[1])
        for row in data
    }

    labels = [
        "Initial Accuracy",
        "Training Accuracy",
        "Test Accuracy"
    ]

    values = [
        metrics[label] * 100
        for label in labels
    ]

    colors = [
        "#94a3b8",
        "#2563eb",
        "#16a34a"
    ]

    content = '''
    <text x="450" y="35" text-anchor="middle"
          font-size="24" font-weight="bold">
        Accuracy Comparison
    </text>
    '''

    # Grid and percentage labels
    for value in range(0, 101, 20):

        y = 400 - value * 3

        content += f'''
        <line x1="90" y1="{y}"
              x2="850" y2="{y}"
              stroke="#e2e8f0"/>

        <text x="78" y="{y + 4}"
              text-anchor="end" font-size="12">
            {value}%
        </text>
        '''

    # Accuracy bars
    for i, value in enumerate(values):

        x = 155 + i * 240
        height = value * 3
        y = 400 - height

        content += f'''
        <rect x="{x}" y="{y}"
              width="140" height="{height}"
              fill="{colors[i]}" rx="4"/>

        <text x="{x + 70}" y="{y - 12}"
              text-anchor="middle"
              font-size="18" font-weight="bold">
            {value:.2f}%
        </text>

        <text x="{x + 70}" y="430"
              text-anchor="middle" font-size="14">
            {labels[i]}
        </text>
        '''

    content += '''
    <text x="25" y="250"
          transform="rotate(-90 25 250)"
          text-anchor="middle" font-size="15">
        Accuracy (%)
    </text>
    '''

    save_svg("accuracy_comparison.svg", content)


def plot_actual_vs_predicted():

    data = np.loadtxt(
        RESULTS_DIR / "test_predictions.csv",
        delimiter=",",
        skiprows=1
    )

    samples = data[:, 0]
    actual = data[:, 1]
    predicted = data[:, 3]

    left, right = 90, 850

    x = left + (
        (samples - samples.min())
        / max(samples.max() - samples.min(), 1)
    ) * (right - left)

    content = '''
    <text x="450" y="35" text-anchor="middle"
          font-size="23" font-weight="bold">
        Actual vs Predicted Classes
    </text>

    <line x1="90" y1="130" x2="850" y2="130"
          stroke="#e2e8f0"/>

    <line x1="90" y1="330" x2="850" y2="330"
          stroke="#e2e8f0"/>

    <text x="65" y="135" font-size="14">1</text>
    <text x="65" y="335" font-size="14">0</text>

    <line x1="90" y1="90" x2="90" y2="370"
          stroke="#334155"/>

    <line x1="90" y1="370" x2="850" y2="370"
          stroke="#334155"/>
    '''

    for i in range(len(samples)):

        actual_y = 330 - actual[i] * 200
        predicted_y = 330 - predicted[i] * 200

        px = x[i]

        # Actual class
        content += f'''
        <circle cx="{px}" cy="{actual_y}"
                r="3" fill="#2563eb"
                opacity="0.65"/>
        '''

        # Predicted class
        content += f'''
        <circle cx="{px}" cy="{predicted_y}"
                r="2" fill="#f59e0b"/>
        '''

        # Highlight incorrect predictions
        if actual[i] != predicted[i]:

            content += f'''
            <line x1="{px}" y1="{actual_y}"
                  x2="{px}" y2="{predicted_y}"
                  stroke="#dc2626"
                  stroke-width="2"
                  stroke-dasharray="5 3"/>

            <circle cx="{px}" cy="{predicted_y}"
                    r="7" fill="none"
                    stroke="#dc2626"
                    stroke-width="2"/>
            '''

    # Sample labels
    for sample in np.linspace(
        samples.min(), samples.max(), 7
    ):

        px = left + (
            (sample - samples.min())
            / max(samples.max() - samples.min(), 1)
        ) * (right - left)

        content += f'''
        <text x="{px}" y="390"
              text-anchor="middle" font-size="12">
            {int(sample)}
        </text>
        '''

    errors = np.sum(actual != predicted)

    content += f'''
    <text x="470" y="420" text-anchor="middle"
          font-size="15">Test Sample</text>

    <text x="25" y="230"
          transform="rotate(-90 25 230)"
          text-anchor="middle" font-size="15">
        Class
    </text>

    <circle cx="200" cy="455" r="5"
            fill="#2563eb"/>

    <text x="212" y="460" font-size="13">
        Actual
    </text>

    <circle cx="350" cy="455" r="5"
            fill="#f59e0b"/>

    <text x="362" y="460" font-size="13">
        Predicted
    </text>

    <circle cx="500" cy="455" r="6"
            fill="none" stroke="#dc2626"
            stroke-width="2"/>

    <text x="512" y="460" font-size="13">
        Incorrect predictions: {errors}
    </text>
    '''

    save_svg("actual_vs_predicted.svg", content)


if __name__ == "__main__":

    print("Generating visualizations...\n")

    plot_training_loss()
    plot_confusion_matrix()
    plot_accuracy_comparison()
    plot_actual_vs_predicted()

    print("\nAll visualizations created successfully.")