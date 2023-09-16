const stripe = require('stripe')(process.env.STRIPE_PRIVATE_KEY)
const orderProcessingModel = require("../models/OrderProcessing")
const orderHistoryModel = require('../models/OrderHistory')
const pantryModel = require('../models/Pantry')
const cartModel = require('../models/Cart')

/* -------------------------------------------------------------------------- */
/*                              Helper Functions                              */
/* -------------------------------------------------------------------------- */

const verifyOrder = async (cartItems) => {
    console.log(cartItems)
    for(let i = 0; i < cartItems.length; i++)
    {
        const pantryItem = await pantryModel.findOne({_id : cartItems[i].itemid})
        if(pantryItem.item_available_qt <= 0 || cartItems[i].quantity <= 0 || pantryItem.item_available_qt < cartItems[i].quantity){
            throw new Error(`Please recheck the Quantity : ${cartItems[i].item_name}, In Stock ${pantryItem.item_available_qt} and In Cart ${cartItems[i].quantity}`)
        }
    }
}

const updatePantry = async(cartItems)=>{
    for(let i = 0; i < cartItems.length; i++)
    {
        const pantryItem = await pantryModel.findOneAndUpdate({_id : cartItems[i].itemid},{$inc : {item_available_qt : -cartItems[i].quantity}})
        //right now panty can go to negative values
    }
}

const createOrderHistory = async (order, stripesSuccessObject)=>{
    const userid = order.userid
    const hist = await orderHistoryModel.findOne({userid : userid})
    console.log(hist)
    if(!hist){
        await orderHistoryModel.create({
            userid : userid,
            stripe_customer : order.stripe_customer,
            order : [],
        })
    }
    const obj = { 
        cartItems : order.cartItems,
        stripe_session_object : order.stripe_session_object,
        stripe_session_success : stripesSuccessObject,
        createdAt : new Date()
    }
    console.log(obj)
    const history = await orderHistoryModel.findOneAndUpdate({userid : userid}, {$push : { orders : obj}})
    console.log(history)
}

const clearCollections = async(userid)=>{
    await orderProcessingModel.deleteOne({userid : userid})
    await cartModel.deleteOne({userid : userid})
}

const fulfillOrder = async (data)=>{
    
    try{
        const processing = await orderProcessingModel.findOne({stripe_customer : data.customer})
        console.log(processing)
        await verifyOrder(processing.cartItems)
        console.log("Order Verified")
        await updatePantry(processing.cartItems)
        console.log("Pantry Updated")
        await createOrderHistory(processing, data)
        console.log("Crated ENtry in Order History")
        await clearCollections(processing.userid)
        console.log("Database Cleared")

    }
    catch(error)
    {
        console.log(error.message)
    }    
}



const createWebHookEvent = async (payload, sig, secret)=>{
    let event;
    try {
        event =  await stripe.webhooks.constructEvent(payload, sig, secret);
    } catch (err) {
        console.log(`Webhook Error: ${err.message}`)
        return res.status(400).send(`Webhook Error: ${err.message}`);
    }
    return event
}

const getLineItems = (cartItems)=>{
    const line_items = []
    cartItems.forEach(item => {
        const str = `
        {
            "price_data": {
                "currency" : "inr",
                "product_data"  : {
                    "name" : "${item.item_name}"
                },
                "unit_amount" : ${item.item_price*100}
            },
            "tax_rates": ["${process.env.TAX_RATE_ID_GST}"],
            "quantity" : ${item.quantity}              
        }`
        line_items.push(JSON.parse(str))
    })
    return line_items
}


const createStripeSession = async (stripeCustomerID, cartItems, checkoutData)=>{
    return await stripe.checkout.sessions.create({
        shipping_address_collection : {"allowed_countries": ["IN"]},
        shipping_options :[
            {
                "shipping_rate_data": {
                "type": "fixed_amount",
                "fixed_amount": {"amount": checkoutData.deliveryCharge*100, "currency": "inr"},
                "display_name": "Shipping Cost"
            },
        }],
        payment_method_types : ['card'],
        mode : 'payment',
        allow_promotion_codes:true,
        customer : stripeCustomerID,
        expires_at : Math. floor(((new Date()).getTime()+(1000*60*30)) / 1000), //30 min
        line_items : getLineItems(cartItems),
        success_url : 'http://127.0.0.1:3000/cart',
        cancel_url : 'http://127.0.0.1:3000/checkout/cancel'
    })
}
/* -------------------------------------------------------------------------- */
/*                                     END                                    */
/* -------------------------------------------------------------------------- */


/* -------------------------------------------------------------------------- */
/*                            Main Driver Function                            */
/* -------------------------------------------------------------------------- */

const cancelEvent = async (req, res)=>{
    try{
        console.log(req.cookies.userid)
        await orderProcessingModel.deleteOne({userid : req.cookies.userid})
        res.redirect('/cart')
    }
    catch(error){
        res.redirect('/cart')
    }
}


const createCheckoutSession = async (req, res)=>{
    const cartItems = req.cartItems;
    const checkoutData = req.checkoutData;
    const stripeCustomerID = req.stripe_customer
    const userid = req.userid
    try{
        // data not yet fetched from the OrderProcessing
        const stripeSession = await createStripeSession(stripeCustomerID, cartItems, checkoutData)
        await orderProcessingModel.updateOne({userid : userid}, {$set : {stripe_session_object : stripeSession}})
        // sending the url
        res.status(201).json({url : stripeSession.url, session : stripeSession})
    }
    catch(error)
    {
        // first remove the order
        await orderProcessingModel.deleteOne({userid : userid})
        res.status(500).json({error : error.message})
    }
}



const stripeWebHookResponse = async (req, res)=>{
    const payload = req.rawBody;
    const sig = req.headers['stripe-signature'];
    const secret = process.env.WEBHOOK_SIGNING_SECRET
    try{
        const event = await createWebHookEvent(payload, sig, secret)
        // Handle the checkout.session.completed event
        if (event.type === 'checkout.session.completed') {
            // Retrieve the session. If you require line items in the response, you may include them by expanding line_items.
            const sessionWithLineItems = await stripe.checkout.sessions.retrieve(
            event.data.object.id,{
                    expand: ['line_items'],
                }
            );
            const lineItems = sessionWithLineItems.line_items;
            console.log(sessionWithLineItems)

            await fulfillOrder(sessionWithLineItems)
        }
        res.status(200).end()
    }
    catch(error)
    {
        console.log(error.message)
    }
}




module.exports = { cancelEvent, createCheckoutSession, stripeWebHookResponse }