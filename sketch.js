let rawdata;
let model;
let data_x = [];
let data_y = [];
let xs, ys;
let labels = [
    1,
    0
]
// Failure. Dataset is limited. Needs a huge amount of epochs and units to process.

function preload() {
    rawdata = loadTable('dataset.csv', 'csv', 'header');
}

function setup() {
    for (i = 0; i < rawdata.rows.length; i++) {
        let rd = rawdata.rows[i].obj
        let col = [
            Number(rd.win_ratio),
            Number(rd.total_accuracy),
            Number(rd.kill_to_death_ratio)
        ];
        data_x.push(col);
        data_y.push(labels.indexOf(Number(rd.VACBanned)));
        
    }
    rawdata = null;
   
    
    xs = tf.tensor2d(data_x);
    const VACTensor = tf.tensor1d(data_y, 'int32');
    ys = tf.oneHot(VACTensor, 2);
    VACTensor.dispose();
    model = tf.sequential();
    let hidden = tf.layers.dense({
        units: 8,
        activation: 'sigmoid',
        inputShape: [3]
    });
    let output = tf.layers.dense({
        units: 2,
        activation: 'softmax'
    });
    model.add(hidden);
    model.add(output);
    const lr = 0.2;
    const optimizer = tf.train.sgd(lr);
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy'
    });
}

function train(eps) {
    console.log('Training started!');
    const options = {
        epochs: eps,
        validationSplit: 0.2,
        shuffle: true
    };
    model.fit(xs, ys, options).then((results) => {
        console.log('Training finished! Loss: '+results.history.loss["0"]);
    });
}


function predict(win_ratio, total_accuracy, kill_to_death_ratio) {
    console.log(vac[(tf.argMax(model.predict(tf.tensor2d([[win_ratio, total_accuracy, kill_to_death_ratio]])), axis=1).dataSync()[0])]);
}