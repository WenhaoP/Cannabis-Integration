from train import *

def prediction(insample, full_dataset, best_val_models):
    """
    Use the best models on the validation set to predict
    """
    descriptions = insample['straindescription']
    descriptions_test = full_dataset['straindescription']
    preds_dict = {}

    for index, row in best_val_models.iterrows():
        label = row['model_name']
        num_filters = row['num_filters'].strip('][').split(', ')
        num_filters = [int(i) for i in num_filters]
        kernel_size = int(row['kernel_size'])
        dilation_rate = int(row['dilation'])
        vocab_size = int(row['vocab_size'])
        embedding_dim = int(row['embedding_dim'])
        maxlen = int(row['maxlen'])
        vs = float(row['validation_set_size'])

        print(f'=== Predicting {label} ===')    
        balanced_f = 'undersampled' in label
        if balanced_f:
            label = label.split('_')[0]
            # Balancing the class distribution
            insample_label = insample[['straindescription', label]]
            class_0 = insample_label[insample_label[label] == 0]
            class_1 = insample_label[insample_label[label] == 1]
            if len(class_0) > len(class_1):
                class_0 = class_0.sample(len(class_1))
            else:
                class_1 = class_1.sample(len(class_0))
            insample_balanced = pd.concat([class_0, class_1], axis=0)
            descriptions = insample_balanced['straindescription']
            Y = insample_balanced[label]
            label = label + '_undersampled'
        else:
            descriptions = insample['straindescription']
            Y = insample[label]

        # Train-test split
        descriptions_train, descriptions_val, y_train, y_val = train_test_split(
            descriptions, Y, test_size=vs, random_state=1000)

        # Tokenize words
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(descriptions_train)
        X_train = tokenizer.texts_to_sequences(descriptions_train)
        X_val = tokenizer.texts_to_sequences(descriptions_val)
        X_test = tokenizer.texts_to_sequences(descriptions_test)

        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        # Pad sequences with zeros
        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)  
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)  

        model = create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen, dilation_rate)

        es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.001, patience=2)
        model.fit(X_train, y_train,
                    epochs=20,
                    verbose=False,
                    batch_size=256,
                    validation_data=(X_val, y_val),
                    callbacks=[es])

        # Evaluate testing set
        predictions = model.predict(X_test)
        predictions = predictions.round()
        
        preds_dict[label] = predictions
        print(f'Positive rate: {np.mean(predictions)}')

    for label in preds_dict:
        full_dataset[f'{label}_labeled'] = preds_dict[label] 
    
    for unpredict_label in (set(FULL_LABELS) - set(LABELS)):
        full_dataset[f"{unpredict_label}_labeled"] = np.zeros(full_dataset.shape[0]).astype(int)

    return full_dataset 